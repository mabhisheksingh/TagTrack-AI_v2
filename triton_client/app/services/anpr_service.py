import cv2
import asyncio
import numpy as np
import supervision as sv
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import structlog
import traceback
import csv
import time

from app.services.paddle_ocr_engine import PaddleOCREngine
from app.services.triton_client import TritonClient
from app.core.config import settings

logger = structlog.get_logger(__name__)

def calculate_blur(image: np.ndarray) -> float:
    """Calculate blur score using Laplacian variance.
    Lower values indicate more blur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


class ANPRService:
    # def __init__(self, triton_client: TritonClient, ocr_service: OCRService):
    def __init__(self, triton_client: TritonClient, ocr_service: PaddleOCREngine):

        self.triton_client = triton_client
        self.ocr_service = ocr_service
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # Load class names and OCR class IDs from settings
        self.class_names = settings.class_names_list
        self.ocr_class_ids = set(settings.ocr_class_ids_list)
        
        logger.info("anpr_service_initialized",
                        class_names=self.class_names,
                        ocr_class_ids=list(self.ocr_class_ids))

    def _process_frame_after_inference(
        self, frame: np.ndarray, detections_raw: np.ndarray, preprocess_meta: dict
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Processes a single frame using pre-computed detections."""

        try:
            logger.debug(
                "detections_raw",
                shape=detections_raw.shape,
                meta=preprocess_meta,
            )
            
            if detections_raw.ndim == 3 and detections_raw.shape[0] == 1:
                detections_raw = detections_raw[0]

            # The model returns 5 columns: [cx, cy, w, h, confidence]
            if detections_raw.shape[1] != 5:
                logger.error("Model output has unexpected shape", shape=detections_raw.shape, expected_columns=5)
                return frame, []

            # Filter by confidence
            confidences = detections_raw[:, 4]
            valid_mask = confidences >= settings.confidence_threshold
            valid_detections = detections_raw[valid_mask]
            
            if len(valid_detections) > 0:
                img_h, img_w = frame.shape[:2]

                input_h, input_w = preprocess_meta.get("input_size", (img_h, img_w))
                x_offset = preprocess_meta.get("x_offset", 0)
                y_offset = preprocess_meta.get("y_offset", 0)
                scale = max(preprocess_meta.get("scale", 1.0), 1e-6)

                # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
                cx = valid_detections[:, 0] * input_w
                cy = valid_detections[:, 1] * input_h
                w = valid_detections[:, 2] * input_w
                h = valid_detections[:, 3] * input_h

                x1 = (cx - w / 2 - x_offset) / scale
                y1 = (cy - h / 2 - y_offset) / scale
                x2 = (cx + w / 2 - x_offset) / scale
                y2 = (cy + h / 2 - y_offset) / scale

                x1 = np.clip(x1, 0, img_w)
                y1 = np.clip(y1, 0, img_h)
                x2 = np.clip(x2, 0, img_w)
                y2 = np.clip(y2, 0, img_h)

                valid_boxes_mask = (x2 > x1) & (y2 > y1)
                if not np.any(valid_boxes_mask):
                    detections = sv.Detections.empty()
                else:
                    xyxy = np.stack([x1, y1, x2, y2], axis=1)[valid_boxes_mask]
                    confidences_filtered = valid_detections[valid_boxes_mask, 4]

                    # Manually assign class_id as 0 since the model is single-class
                    class_ids = np.zeros(len(xyxy), dtype=int)

                    detections = sv.Detections(
                        xyxy=xyxy,
                        confidence=confidences_filtered,
                        class_id=class_ids
                    )
            else:
                detections = sv.Detections.empty()
            
            logger.info("Filtered detections", count=len(detections))
                
            detections = self.tracker.update_with_detections(detections)
            detection_records = []
            ocr_time_total = 0.0

            for order_idx, (class_id, tracker_id, box, confidence) in enumerate(
                zip(
                    detections.class_id,
                    detections.tracker_id,
                    detections.xyxy,
                    detections.confidence,
                )
            ):
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                base_label = f"#{tracker_id} {class_name}"

                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                number_plate_crop = frame[y1:y2, x1:x2]

                if number_plate_crop.size > 0:
                    plate_blur_score = float(calculate_blur(number_plate_crop))
                else:
                    plate_blur_score = 0.0

                result_item = {
                    "tracker_id": int(tracker_id),
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "bbox": [float(x) for x in box],
                    "blur_score": plate_blur_score,
                    "ocr_confidence": 0.0,
                }

                detection_records.append(
                    {
                        "order": order_idx,
                        "base_label": base_label,
                        "label": base_label,
                        "box": box,
                        "confidence": float(confidence),
                        "tracker_id": tracker_id,
                        "blur_score": plate_blur_score,
                        "result_item": result_item,
                        "crop": number_plate_crop,
                        "ocr_confidence": 0.0,
                    }
                )

            # Prioritize higher-quality detections before running OCR so multiple boxes per frame
            # get processed in order of blur score (sharpness) and model confidence.
            sorted_records = sorted(
                detection_records,
                key=lambda record: (record["blur_score"], record["confidence"]),
                reverse=True,
            )

            for record in sorted_records:
                crop = record["crop"]
                confidence = record["confidence"]
                blur_score = record["blur_score"]

                should_run_ocr = (
                    settings.enable_ocr
                    and crop.size > 0
                    and confidence >= settings.ocr_trigger_confidence_threshold
                    and blur_score >= settings.plate_blur_threshold
                )

                if should_run_ocr:
                    logger.info("Running ocr ", crop_image_size=crop.size)
                    ocr_start = time.perf_counter()
                    try:
                        ocr_result = self.ocr_service.recognize(crop)

                        final_confidence = 0.0
                        if isinstance(ocr_result, list):
                            text_parts = []
                            confidences = []
                            for item in ocr_result:
                                text_value = item.get("text", "").strip()
                                if text_value:
                                    text_parts.append(text_value)
                                conf_value = item.get("confidence")
                                if conf_value is not None:
                                    try:
                                        confidences.append(float(conf_value))
                                    except (TypeError, ValueError):
                                        continue
                            final_text = " ".join(text_parts).strip()
                            if confidences:
                                final_confidence = float(sum(confidences) / len(confidences))
                        else:
                            final_text = (ocr_result or "").strip()

                        if final_text:
                            record["label"] = f"{record['base_label']} {final_text}"
                            record["result_item"]["ocr_text"] = final_text
                            record["result_item"]["ocr_confidence"] = final_confidence
                            record["ocr_confidence"] = final_confidence
                        else:
                            logger.debug(
                                "ocr_empty",
                                tracker_id=record["tracker_id"],
                                bbox=record["box"].tolist(),
                                blur_score=blur_score,
                                confidence=confidence,
                            )
                    except Exception as e:
                        logger.error(
                            "ocr_error",
                            tracker_id=record["tracker_id"],
                            error=str(e),
                            blur_score=blur_score,
                            confidence=confidence,
                        )
                    finally:
                        ocr_duration = time.perf_counter() - ocr_start
                        ocr_time_total += ocr_duration
                        logger.debug(
                            "ocr_latency",
                            tracker_id=record["tracker_id"],
                            elapsed_ms=round(ocr_duration * 1000, 3),
                        )
                else:
                    logger.debug(
                        "ocr_skipped",
                        tracker_id=record["tracker_id"],
                        bbox=record["box"].tolist(),
                        blur_score=blur_score,
                        confidence=confidence,
                        reason="insufficient_blur_or_confidence",
                    )

            # Restore original detection order for annotation overlays/results
            detection_records.sort(key=lambda record: record["order"])
            labels = [record["label"] for record in detection_records]
            results = [record["result_item"] for record in detection_records]

            annotated_frame = self.box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

            logger.debug(
                "frame_processing_time",
                ocr_ms=round(ocr_time_total * 1000, 3),
                detections=len(detection_records),
            )
            return annotated_frame, results
        except Exception as e:
            logger.error("frame_post_processing_error", error=str(e), traceback=traceback.format_exc())
            return frame, []

    async def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Processes a single frame by running inference and then post-processing."""
        logger.info("Processing frame", frame_size=frame.shape, frame_idx=frame_idx)
        try:
            # Assuming infer for single image still works with `image` kwarg
            detections_raw, preprocess_meta = await self.triton_client.infer(image=frame, frame_idx=frame_idx)
            return self._process_frame_after_inference(frame, detections_raw, preprocess_meta)
        except Exception as e:
            logger.error("frame_processing_error", frame_idx=frame_idx, error=str(e), traceback=traceback.format_exc())
            return frame, []

    async def process_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            return await self.process_frame(image, frame_idx=0)
        except Exception as e:
            logger.error("image_processing_error", path=image_path, error=str(e))
            raise

    async def process_video(
        self,
        video_path: str,
        output_dir: str,
        save_csv: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file and save annotated output.
        Processes ALL frames but filters out blurry ones.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save output video
            save_csv: Whether to save CSV with detection results
            request_id: request id per request
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video : {video_path}")
        try:
            video_info = sv.VideoInfo.from_video_path(video_path)
            frames_generator = sv.get_video_frames_generator(video_path)
            
            # Use absolute path from settings
            base_output_dir = Path(output_dir).resolve() if output_dir else settings.data_output_dir
            base_output_dir.mkdir(parents=True, exist_ok=True)
            folder_name = request_id or Path(video_path).stem
            request_folder = base_output_dir / folder_name
            request_folder.mkdir(parents=True, exist_ok=True)
            output_path = str(request_folder / Path(video_path).name)
            
            # Create a directory for debug frames
            debug_frames_dir = request_folder / "debug_frames"
            debug_frames_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("processing_video",
                        path=video_path,
                        fps=video_info.fps,
                        total_frames=video_info.total_frames,
                        blur_threshold=settings.blur_threshold,
                        confidence_threshold=settings.confidence_threshold,
                        output_path=output_path,
                        output_dir=str(request_folder),
                        class_names=settings.class_names_list,
                        save_csv=save_csv
                        )
            
            all_detections = []
            frame_count = 0
            processed_count = 0
            frames_batch = []
            frame_indices_batch = []
            
            with sv.VideoSink(output_path, video_info) as sink:
                for frame_idx, frame in enumerate(frames_generator):
                    frames_batch.append(frame)
                    frame_indices_batch.append(frame_idx)

                    is_last_frame = (frame_idx == video_info.total_frames - 1)

                    # If batch is not full and it's not the last frame, continue accumulating
                    if len(frames_batch) < settings.batch_size and not is_last_frame:
                        continue
                    
                    if not frames_batch:
                        continue

                    logger.info(f"Processing batch of {len(frames_batch)} frames", start_frame=frame_indices_batch[0], end_frame=frame_indices_batch[-1])
                    
                    try:
                        # Perform batch inference using asyncio.gather for concurrent requests
                        inference_tasks = [
                            self.triton_client.infer(image=frame, frame_idx=idx) 
                            for frame, idx in zip(frames_batch, frame_indices_batch)
                        ]
                        results = await asyncio.gather(*inference_tasks)
                        detections_batch = [r[0] for r in results]
                        metas_batch = [r[1] for r in results]

                        # Process each frame in the batch with its corresponding detection result
                        for i, (single_frame, detections_raw, meta) in enumerate(zip(frames_batch, detections_batch, metas_batch)):
                            current_frame_idx = frame_indices_batch[i]

                            # Save debug frames
                            if current_frame_idx < 5:
                                debug_frame_path = debug_frames_dir / f"frame_{current_frame_idx:04d}.jpg"
                                cv2.imwrite(str(debug_frame_path), single_frame)
                                logger.info("Saved debug frame", path=str(debug_frame_path))

                            annotated_frame, results = self._process_frame_after_inference(single_frame, detections_raw, meta)

                            for result in results:
                                result['frame'] = current_frame_idx
                            
                            all_detections.extend(results)
                            processed_count += 1
                            sink.write_frame(annotated_frame)

                    except Exception as e:
                        logger.error(
                            "video_batch_error",
                            frames=frame_indices_batch,
                            error=str(e),
                            traceback=traceback.format_exc()
                        )
                        # On error, write original frames to sink
                        for f in frames_batch:
                            sink.write_frame(f)
                    
                    frame_count += len(frames_batch)
                    # Clear the batch for the next set of frames
                    frames_batch.clear()
                    frame_indices_batch.clear()

            logger.info("video_processing_complete",
                           total_frames=frame_count,
                           processed_frames=processed_count,
                           detections=len(all_detections))
            logger.info("Now start saving csv if requested ", save_csv=save_csv)
            # Save CSV if requested
            csv_path = None
            summary_csv_path = None
            if save_csv and all_detections:
                csv_path = str(Path(output_path).with_suffix('.csv'))
                try:
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=settings.CSV_FRAME_HEADER)
                        writer.writeheader()
                        for det in all_detections:
                            writer.writerow({
                                'frame': det.get('frame', 0),
                                'track_id': det.get('tracker_id', -1),
                                'plate_text': det.get('ocr_text', ''),
                                'confidence': det.get('confidence', 0.0),
                                'confidence_ocr': det.get('ocr_confidence', 0.0)
                            })
                    logger.info("csv_saved", path=csv_path)
                except Exception as e:
                    logger.error("csv_save_error", error=str(e))
                else:
                    summary_csv_path = self._write_track_summary_csv(all_detections, Path(csv_path))
            
            result = {
                "video_path": video_path,
                "output_path": output_path,
                "csv_path": csv_path,
                "summary_csv_path": summary_csv_path,
                "total_frames": frame_count,
                "processed_frames": processed_count,
                "total_detections": len(all_detections),
                "detections": all_detections
            }

            logger.info("video_processing_complete",
                           total_frames=frame_count,
                           processed_frames=processed_count,
                           detections=len(all_detections))

            return result
            
        except Exception as e:
            logger.error("video_processing_error",
                            path=video_path,
                            error=str(e),
                            traceback=traceback.format_exc())
            raise

    def _aggregate_track_votes(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        track_votes: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"count": 0, "confidence_sum": 0.0, "ocr_confidence_sum": 0.0})
        )

        for det in detections or []:
            track_id_raw = det.get("tracker_id")
            track_id = int(track_id_raw) if track_id_raw is not None else -1
            plate_text = (det.get("ocr_text") or "").strip()
            if not plate_text:
                continue

            entry = track_votes[track_id][plate_text]
            entry["count"] += 1
            entry["confidence_sum"] += float(det.get("confidence", 0.0))
            entry["ocr_confidence_sum"] += float(det.get("ocr_confidence", 0.0))

        rows: List[Dict[str, Any]] = []
        for track_id, votes in track_votes.items():
            def vote_key(item):
                stats = item[1]
                count = stats["count"]
                avg_ocr = stats["ocr_confidence_sum"] / count if count else 0.0
                avg_conf = stats["confidence_sum"] / count if count else 0.0
                return (count, avg_ocr, avg_conf)

            best_text, stats = max(votes.items(), key=vote_key)
            count = stats["count"]
            avg_conf = stats["confidence_sum"] / count if count else 0.0
            avg_ocr = stats["ocr_confidence_sum"] / count if count else 0.0

            rows.append(
                {
                    "track_id": track_id,
                    "plate_text": best_text,
                    "votes": count,
                    "avg_confidence": round(avg_conf, 4),
                    "avg_confidence_ocr": round(avg_ocr, 4),
                }
            )

        return rows

    def _write_track_summary_csv(
        self,
        detections: List[Dict[str, Any]],
        frame_csv_path: Path,
    ) -> Optional[str]:
        rows = self._aggregate_track_votes(detections)
        if not rows:
            logger.info("track_summary_skipped", reason="no_ocr_results")
            return None

        summary_path = frame_csv_path.with_name(f"{frame_csv_path.stem}_track_summary.csv")

        try:
            with open(summary_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=settings.CSV_TRACK_SUMMARY_HEADER)
                writer.writeheader()
                writer.writerows(rows)

            logger.info(
                "track_summary_csv_saved",
                path=str(summary_path),
                track_count=len(rows),
            )
            return str(summary_path)
        except Exception as e:
            logger.error("track_summary_csv_error", error=str(e), path=str(summary_path))
            return None

    def _build_source_summary_rows(self, source: str, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = self._aggregate_track_votes(detections)
        for row in rows:
            row["source"] = source
        return rows

    def _write_csv_rows(
        self,
        csv_path: Path,
        rows: List[Dict[str, Any]],
        header: List[str],
    ) -> Optional[str]:
        if not rows:
            return None

        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
            logger.info("csv_saved", path=str(csv_path), rows=len(rows))
            return str(csv_path)
        except Exception as e:
            logger.error("csv_write_error", path=str(csv_path), error=str(e))
            return None

    async def process_videos_and_images(
        self,
        *,
        input_folder: str,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Processing videos and images : {input_folder}")
        try:
            input_folder = Path(input_folder)
            effective_request_id = request_id or f"request_{int(time.time()*1000)}"
            request_folder = settings.data_output_dir / effective_request_id
            request_folder.mkdir(parents=True, exist_ok=True)

            consolidated_rows: List[Dict[str, Any]] = []
            image_rows: List[Dict[str, Any]] = []
            video_outputs: List[Dict[str, Any]] = []

            for file in input_folder.iterdir():
                if not file.is_file():
                    continue

                extension = file.suffix.lower()
                if extension in settings.video_extensions:
                    video_result = await self.process_video(
                        str(file),
                        str(settings.data_output_dir),
                        save_csv=True,
                        request_id=effective_request_id,
                    )
                    video_outputs.append(video_result)
                    consolidated_rows.extend(
                        self._build_source_summary_rows(file.name, video_result.get("detections", []))
                    )
                elif extension in settings.image_extensions:
                    annotated_frame, detections = await self.process_image(str(file))
                    output_path = request_folder / file.name
                    cv2.imwrite(str(output_path), annotated_frame)
                    rows = self._build_source_summary_rows(file.name, detections)
                    consolidated_rows.extend(rows)
                    image_rows.extend(rows)

            consolidated_csv = self._write_csv_rows(
                request_folder / f"{effective_request_id}_consolidated.csv",
                consolidated_rows,
                settings.CSV_CONSOLIDATED_HEADER,
            )
            images_csv = self._write_csv_rows(
                request_folder / f"{effective_request_id}_images.csv",
                image_rows,
                settings.CSV_CONSOLIDATED_HEADER,
            )

            return {
                "request_id": effective_request_id,
                "output_dir": str(request_folder),
                "videos": video_outputs,
                "consolidated_csv": consolidated_csv,
                "images_csv": images_csv,
            }
        except Exception as e:
            logger.error(f"Error processing videos and images : {input_folder}", error=str(e))
            raise e


def get_anpr_service() -> ANPRService:
    triton_client = TritonClient(
        server_url=settings.triton_server_url,
        model_name=settings.triton_model_name
    )
    ocr_service = PaddleOCREngine()
    # ocr_service = OCRService()
    return ANPRService(triton_client=triton_client, ocr_service=ocr_service)
