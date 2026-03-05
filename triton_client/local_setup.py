#!/usr/bin/env python3
"""
Local setup script for ANPR processing.
Scans input directory, processes all images, and saves results to output directory.
"""
import argparse
import json
import time
from pathlib import Path
from typing import List
import structlog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from app.services.triton_client import TritonClient
from app.services.paddle_ocr_engine import PaddleOCREngine
from app.services.anpr_service import ANPRService
from app.core.config import settings
from app.core.logging import configure_logging
import cv2


class FileHandler(FileSystemEventHandler):
    """Handler for new image and video files in the input directory."""
    
    def __init__(self, anpr_service: ANPRService, output_folder: str, save_json: bool = False):
        self.anpr_service = anpr_service
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.save_json = save_json
        self.logger = structlog.get_logger(__name__)
        self.image_extensions = set(ext.lower() for ext in settings.image_extensions_list)
        self.video_extensions = set(ext.lower() for ext in settings.video_extensions_list)
    
    def on_created(self, event):
        """Process newly created image or video files."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        file_ext = file_path.suffix.lower()
        
        # Check if it's an image
        if file_ext in self.image_extensions:
            time.sleep(0.5)  # Wait for file to be fully written
            self.logger.info("new_image_detected", file=str(file_path))
            self.process_single_image(str(file_path))
        # Check if it's a video and video processing is enabled
        elif file_ext in self.video_extensions and settings.enable_video_detection:
            time.sleep(1.0)  # Videos need more time to be fully written
            self.logger.info("new_video_detected", file=str(file_path))
            self.process_single_video(str(file_path))
        else:
            return
        
    
    def process_single_image(self, file_path: str):
        """Process a single image file."""
        try:
            import cv2
            import numpy as np
            
            file_path_obj = Path(file_path)
            
            # Process the image
            annotated_frame, results = self.anpr_service.process_image(file_path)
            
            # Save annotated image
            output_file = self.output_folder / file_path_obj.name
            cv2.imwrite(str(output_file), annotated_frame)
            
            self.logger.info("file_processed",
                           input=str(file_path_obj.name),
                           output=str(output_file),
                           detections=len(results))
            
            # Save JSON if requested
            if self.save_json:
                json_file = self.output_folder / f"{file_path_obj.stem}_result.json"
                with open(json_file, 'w') as f:
                    json.dump({
                        "image_path": str(file_path),
                        "output_path": str(output_file),
                        "detections": results
                    }, f, indent=2)
            
            # Print results
            print(f"\n✓ Processed: {file_path_obj.name}")
            for det in results:
                print(f"  - {det['class_name']} (conf: {det['confidence']:.2f})", end="")
                if 'ocr_text' in det:
                    print(f" - Plate: {det['ocr_text']}")
                else:
                    print()
        
        except Exception as e:
            self.logger.error("image_processing_error", file=file_path, error=str(e))
            print(f"✗ Error processing {Path(file_path).name}: {str(e)}")
    
    def process_single_video(self, file_path: str):
        """Process a single video file."""
        try:
            file_path_obj = Path(file_path)
            output_file = self.output_folder / file_path_obj.name
            
            # Process the video
            result = self.anpr_service.process_video(
                file_path,
                str(output_file),
                save_csv=self.save_json
            )
            
            # Print results
            print(f"\n✓ Processed: {file_path_obj.name}")
            print(f"  - Frames: {result['processed_frames']}/{result['total_frames']}")
            print(f"  - Detections: {result['total_detections']}")
            if result.get('csv_path'):
                print(f"  - CSV: {Path(result['csv_path']).name}")
        
        except Exception as e:
            self.logger.error("video_processing_error", file=file_path, error=str(e))
            print(f"✗ Error processing {Path(file_path).name}: {str(e)}")


def process_existing_files(input_folder: str, output_folder: str, 
                          anpr_service: ANPRService, save_json: bool = False):
    """Process all existing files (images and videos) in the input folder."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = structlog.get_logger(__name__)
    image_extensions = set(ext.lower() for ext in settings.image_extensions_list)
    video_extensions = set(ext.lower() for ext in settings.video_extensions_list)
    
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    video_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ] if settings.enable_video_detection else []
    
    if not image_files and not video_files:
        logger.warning("no_files_found", folder=input_folder)
        print(f"\n⚠ No images or videos found in {input_folder}")
        return
    
    total_files = len(image_files) + len(video_files)
    logger.info("processing_existing_files", images=len(image_files), videos=len(video_files))
    print(f"\n{'='*60}")
    print(f"Processing {len(image_files)} images and {len(video_files)} videos...")
    print(f"{'='*60}\n")
    
    # Process in batches
    batch_size = settings.batch_size
    all_results = []
    
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]
        batch_paths = [str(f) for f in batch]
        
        batch_results = anpr_service.process_batch(batch_paths)
        
        # Save annotated images
        import cv2
        import numpy as np
        import supervision as sv
        
        for result in batch_results:
            image_path = result["image_path"]
            image = cv2.imread(image_path)
            
            if image is not None:
                detections_list = result["detections"]
                
                if detections_list:
                    xyxy = np.array([d["bbox"] for d in detections_list])
                    confidence = np.array([d["confidence"] for d in detections_list])
                    class_id = np.array([d["class_id"] for d in detections_list])
                    
                    detections = sv.Detections(
                        xyxy=xyxy,
                        confidence=confidence,
                        class_id=class_id
                    )
                    
                    labels = []
                    for d in detections_list:
                        label = f"{d['class_name']}"
                        if "ocr_text" in d:
                            label += f" - {d['ocr_text']}"
                        labels.append(label)
                    
                    box_annotator = sv.BoxAnnotator()
                    label_annotator = sv.LabelAnnotator()
                    
                    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
                    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                else:
                    annotated = image
                
                output_file = output_path / Path(image_path).name
                cv2.imwrite(str(output_file), annotated)
                result["output_path"] = str(output_file)
                
                # Print results
                print(f"✓ Processed: {Path(image_path).name}")
                for det in detections_list:
                    print(f"  - {det['class_name']} (conf: {det['confidence']:.2f})", end="")
                    if 'ocr_text' in det:
                        print(f" - Plate: {det['ocr_text']}")
                    else:
                        print()
        
        all_results.extend(batch_results)
    
    # Save JSON summary if requested
    if save_json:
        json_file = output_path / "batch_results.json"
        with open(json_file, 'w') as f:
            json.dump({
                "processed": len(all_results),
                "results": all_results
            }, f, indent=2)
        print(f"\n📄 Results saved to: {json_file}")
    
    # Process videos if enabled
    video_results = []
    if video_files and settings.enable_video_detection:
        print(f"\n{'='*60}")
        print(f"Processing {len(video_files)} videos...")
        print(f"{'='*60}\n")
        
        for video_file in video_files:
            try:
                output_video = output_path / video_file.name
                result = anpr_service.process_video(
                    str(video_file),
                    str(output_video),
                    save_csv=save_json
                )
                video_results.append(result)
                
                print(f"✓ Processed: {video_file.name}")
                print(f"  - Frames: {result['processed_frames']}/{result['total_frames']}")
                print(f"  - Detections: {result['total_detections']}")
                if result.get('csv_path'):
                    print(f"  - CSV: {Path(result['csv_path']).name}")
                    
            except Exception as e:
                logger.error("video_processing_error", file=str(video_file), error=str(e))
                print(f"✗ Error processing {video_file.name}: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"  - Images: {len(all_results)}")
    print(f"  - Videos: {len(video_results)}")
    print(f"Output directory: {output_folder}")
    print(f"{'='*60}\n")


def main():
    configure_logging(settings.log_level, settings.app_env)
    logger = structlog.get_logger(__name__)
    
    parser = argparse.ArgumentParser(description="ANPR Local Setup - Process images from input directory")
    parser.add_argument(
        "--input-folder",
        default=settings.input_folder,
        help=f"Path to input folder containing images (default: {settings.input_folder})"
    )
    parser.add_argument(
        "--output-folder",
        default=settings.output_folder,
        help=f"Path to output folder for annotated images (default: {settings.output_folder})"
    )
    parser.add_argument(
        "--server-url",
        default=settings.triton_server_url,
        help=f"Triton server URL (default: {settings.triton_server_url})"
    )
    parser.add_argument(
        "--model-name",
        default=settings.triton_model_name,
        help=f"Triton model name (default: {settings.triton_model_name})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.batch_size,
        help=f"Batch size for processing (default: {settings.batch_size})"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch input folder for new files (continuous mode)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detection results as JSON files"
    )
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    Path(args.input_folder).mkdir(parents=True, exist_ok=True)
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    
    logger.info("starting_local_setup",
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                server_url=args.server_url,
                model_name=args.model_name,
                batch_size=args.batch_size,
                watch_mode=args.watch)
    
    print(f"\n{'='*60}")
    print(f"ANPR Local Setup")
    print(f"{'='*60}")
    print(f"Input folder:  {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Triton server: {args.server_url}")
    print(f"Model name:    {args.model_name}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Watch mode:    {'Enabled' if args.watch else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # Initialize services
    triton_client = TritonClient(server_url=args.server_url, model_name=args.model_name)
    ocr_service = PaddleOCREngine()
    anpr_service = ANPRService(triton_client=triton_client, ocr_service=ocr_service)
    
    settings.batch_size = args.batch_size
    
    # Process existing files first
    process_existing_files(args.input_folder, args.output_folder, anpr_service, args.save_json)
    
    # Watch for new files if requested
    if args.watch:
        file_types = "images"
        if settings.enable_video_detection:
            file_types = "images and videos"
        print(f"\n👁 Watch mode enabled - monitoring for new {file_types}...")
        print("Press Ctrl+C to stop\n")
        
        event_handler = FileHandler(anpr_service, args.output_folder, args.save_json)
        observer = Observer()
        observer.schedule(event_handler, args.input_folder, recursive=False)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\n\n✓ Stopped watching. Goodbye!")
        
        observer.join()


if __name__ == "__main__":
    main()
