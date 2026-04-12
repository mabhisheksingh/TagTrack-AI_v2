import numpy as np
import cv2
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
from pathlib import Path


class SimpleTritonVehicleApp:
    def __init__(self, server_url="127.0.0.1:9001"):
        self.client = grpcclient.InferenceServerClient(url=server_url)
        print(f"Connected to Triton at {server_url}")

    def preprocess(self, img, target_size=(640, 640)):
        """Resize and normalize image for RT-DETR."""
        h, w = img.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((target_size[1], target_size[0], 3), 114, dtype=np.uint8)

        # Center the image (Letterbox)
        x_off, y_off = (target_size[0] - nw) // 2, (target_size[1] - nh) // 2
        canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

        # Convert to CHW and Normalize
        blob = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, axis=0), {"scale": scale, "x_off": x_off, "y_off": y_off}

    def postprocess(self, raw_data, meta, original_shape, conf_thresh=0.4):
        """Handle both RT-DETR [1, 300, 19] and YOLO [1, 300, 6] output formats."""
        detections = raw_data[0]  # Remove batch dim -> [300, N]
        print("detections shape:", detections.shape, "first row:", detections[0])
        results = []
        h_orig, w_orig = original_shape
        
        num_cols = detections.shape[1]

        for row in detections:
            cx, cy, w, h = row[:4]
            
            # Detect format: YOLO [cx,cy,w,h,conf,class] vs RT-DETR [cx,cy,w,h,score1,score2,...]
            if num_cols == 6:
                # YOLO format: [cx, cy, w, h, confidence, class_id]
                # Coordinates are ABSOLUTE pixels in 640x640 space
                conf = row[4]
                class_id = int(row[5])
                
                if conf >= conf_thresh:
                    # Debug: print first detection details
                    if len(results) == 0:
                        print(f"DEBUG first detection:")
                        print(f"  Raw: cx={cx:.2f}, cy={cy:.2f}, w={w:.2f}, h={h:.2f}")
                        print(f"  Meta: x_off={meta['x_off']}, y_off={meta['y_off']}, scale={meta['scale']:.4f}")
                    
                    # YOLO outputs absolute coords in 640x640, just remove padding and scale
                    x1 = (cx - w / 2 - meta['x_off']) / meta['scale']
                    y1 = (cy - h / 2 - meta['y_off']) / meta['scale']
                    x2 = (cx + w / 2 - meta['x_off']) / meta['scale']
                    y2 = (cy + h / 2 - meta['y_off']) / meta['scale']
                    
                    if len(results) == 0:
                        print(f"  Computed: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
                        print(f"  Box size: w={x2-x1:.2f}, h={y2-y1:.2f}")
                    
                    results.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": float(conf),
                        "class_id": class_id
                    })
            else:
                # RT-DETR format: [cx, cy, w, h, class_score1, class_score2, ...]
                # Coordinates are NORMALIZED (0-1), need to multiply by 640
                scores = row[4:]
                conf = np.max(scores)
                class_id = int(np.argmax(scores))

                if conf >= conf_thresh:
                    # Convert normalized to 640x640 space, then remove padding and scale
                    x1 = ((cx - w / 2) * 640 - meta['x_off']) / meta['scale']
                    y1 = ((cy - h / 2) * 640 - meta['y_off']) / meta['scale']
                    x2 = ((cx + w / 2) * 640 - meta['x_off']) / meta['scale']
                    y2 = ((cy + h / 2) * 640 - meta['y_off']) / meta['scale']

                    results.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": float(conf),
                        "class_id": class_id
                    })
        return results

    def run(self, model_name, img_path):
        img = cv2.imread(img_path)
        if img is None: return print("Image not found!")

        # 1. Prepare Data
        input_data, meta = self.preprocess(img)

        # 2. Triton Inference
        inputs = [grpcclient.InferInput("images", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput("output0")]

        response = self.client.infer(model_name, inputs, outputs=outputs)
        raw_results = response.as_numpy("output0")

        # 3. Process & Draw
        detections = self.postprocess(raw_results, meta, img.shape[:2])
        print(f"Found {len(detections)} vehicles.")

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Vehicle: {det['conf']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Save
        out_name = "detected_vehicles.jpg"
        cv2.imwrite(out_name, img)
        print(f"Saved result to {out_name}")


if __name__ == "__main__":
    app = SimpleTritonVehicleApp()
    BASE_DIR = Path(__file__).resolve().parent
    image_path = BASE_DIR / "frame_0000.jpg"
    # Replace with your image filename
    app.run("vehicle_detection_rt_detr", img_path=image_path)
    # app.run("object_detection_yolo26x", img_path=image_path)

    