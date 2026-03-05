# Dual-Model ANPR Architecture Design

## Overview
Run **vehicle detection** and **license plate detection** models in parallel, then combine results to get:
- Vehicle bounding boxes with speed estimation
- License plate text associated with each vehicle
- Complete vehicle tracking with plate information

## Architecture

### Current Setup (Single Model)
```
Frame → Plate Detection Model → Detections → OCR → Results
```

### Future Setup (Dual Model)
```
                    ┌─→ Vehicle Model → Vehicle Detections ─┐
Frame ──(parallel)──┤                                        ├─→ Combine → Results
                    └─→ Plate Model → Plate Detections ─────┘
                                            ↓
                                          OCR
```

## Implementation Plan

### Phase 1: Multi-Model Configuration
```bash
# .env configuration
TRITON_MODELS="vehicle_detection,plate_detection"

# Vehicle detection model
VEHICLE_MODEL_NAME="yolov8_vehicle"
VEHICLE_CLASS_NAMES="car,truck,bus,motorcycle,bicycle"
VEHICLE_OCR_CLASS_IDS=""  # No OCR for vehicles

# Plate detection model
PLATE_MODEL_NAME="plate_region_detection_rt_detr"
PLATE_CLASS_NAMES="number_plate"
PLATE_OCR_CLASS_IDS="0"  # Run OCR on plates
```

### Phase 2: Parallel Inference
```python
class MultiModelANPRService:
    def __init__(self, vehicle_client, plate_client, ocr_service):
        self.vehicle_client = vehicle_client
        self.plate_client = plate_client
        self.ocr_service = ocr_service
        self.vehicle_tracker = sv.ByteTrack()
        self.plate_tracker = sv.ByteTrack()
    
    def process_frame(self, frame):
        # Run both models in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            vehicle_future = executor.submit(self.vehicle_client.infer, frame)
            plate_future = executor.submit(self.plate_client.infer, frame)
            
            vehicle_detections = vehicle_future.result()
            plate_detections = plate_future.result()
        
        # Track both
        vehicle_detections = self.vehicle_tracker.update_with_detections(vehicle_detections)
        plate_detections = self.plate_tracker.update_with_detections(plate_detections)
        
        # Associate plates with vehicles
        results = self.associate_plates_with_vehicles(vehicle_detections, plate_detections)
        
        return results
```

### Phase 3: Plate-Vehicle Association
```python
def associate_plates_with_vehicles(self, vehicles, plates):
    """
    Match license plates to vehicles using spatial proximity.
    
    Logic:
    1. For each plate, find the nearest vehicle
    2. Check if plate is inside or near vehicle bbox
    3. Associate plate with vehicle tracker_id
    """
    associations = []
    
    for plate in plates:
        plate_center = get_bbox_center(plate.xyxy)
        
        # Find closest vehicle
        min_distance = float('inf')
        matched_vehicle = None
        
        for vehicle in vehicles:
            if is_plate_inside_vehicle(plate.xyxy, vehicle.xyxy):
                matched_vehicle = vehicle
                break
            
            distance = euclidean_distance(plate_center, get_bbox_center(vehicle.xyxy))
            if distance < min_distance:
                min_distance = distance
                matched_vehicle = vehicle
        
        if matched_vehicle:
            associations.append({
                'vehicle_tracker_id': matched_vehicle.tracker_id,
                'vehicle_class': matched_vehicle.class_name,
                'plate_tracker_id': plate.tracker_id,
                'plate_text': plate.ocr_text,
                'confidence': plate.confidence
            })
    
    return associations
```

### Phase 4: Speed Estimation
```python
class SpeedEstimator:
    def __init__(self, fps, pixel_to_meter_ratio):
        self.fps = fps
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self.vehicle_positions = {}  # {tracker_id: [(frame, position), ...]}
    
    def estimate_speed(self, tracker_id, current_position, frame_number):
        if tracker_id not in self.vehicle_positions:
            self.vehicle_positions[tracker_id] = []
        
        self.vehicle_positions[tracker_id].append((frame_number, current_position))
        
        # Calculate speed using last N positions
        positions = self.vehicle_positions[tracker_id][-10:]  # Last 10 frames
        
        if len(positions) < 2:
            return 0.0
        
        # Calculate displacement
        start_frame, start_pos = positions[0]
        end_frame, end_pos = positions[-1]
        
        pixel_distance = euclidean_distance(start_pos, end_pos)
        meter_distance = pixel_distance * self.pixel_to_meter_ratio
        
        time_elapsed = (end_frame - start_frame) / self.fps
        speed_mps = meter_distance / time_elapsed if time_elapsed > 0 else 0
        speed_kmph = speed_mps * 3.6
        
        return speed_kmph
```

## Configuration Examples

### Example 1: Dual Model (Vehicle + Plate)
```bash
# Enable dual model mode
USE_DUAL_MODEL=true

# Vehicle detection
VEHICLE_MODEL_NAME="yolov8_vehicle"
VEHICLE_MODEL_URL="127.0.0.1:8001"
VEHICLE_CLASS_NAMES="car,truck,bus,motorcycle"

# Plate detection
PLATE_MODEL_NAME="plate_region_detection_rt_detr"
PLATE_MODEL_URL="127.0.0.1:8001"
PLATE_CLASS_NAMES="number_plate"
PLATE_OCR_CLASS_IDS="0"

# Speed estimation
ENABLE_SPEED_ESTIMATION=true
PIXEL_TO_METER_RATIO=0.05  # Calibrate based on camera setup
```

### Example 2: Single Model (Current)
```bash
USE_DUAL_MODEL=false
TRITON_MODEL_NAME="plate_region_detection_rt_detr"
MODEL_CLASS_NAMES="number_plate"
OCR_CLASS_IDS="0"
```

## Output Format

### CSV Output with Dual Model
```csv
frame,vehicle_tracker_id,vehicle_class,vehicle_speed_kmph,plate_tracker_id,plate_text,plate_confidence,blur_score
0,1,car,45.2,5,ABC123,0.92,150.3
0,2,truck,38.7,6,XYZ789,0.88,142.1
1,1,car,46.1,5,ABC123,0.91,148.7
```

### JSON Output
```json
{
  "frame": 0,
  "vehicles": [
    {
      "tracker_id": 1,
      "class": "car",
      "bbox": [100, 200, 300, 400],
      "speed_kmph": 45.2,
      "plate": {
        "tracker_id": 5,
        "text": "ABC123",
        "confidence": 0.92,
        "bbox": [150, 320, 250, 360]
      }
    }
  ]
}
```

## Benefits

1. **Parallel Processing**: 2x faster than sequential
2. **Better Accuracy**: Specialized models for each task
3. **Speed Estimation**: Track vehicles across frames
4. **Complete Context**: Vehicle type + plate + speed
5. **Flexible**: Can disable either model independently

## Migration Path

1. ✅ **Current**: Single plate detection model working
2. 🔄 **Next**: Add vehicle detection model configuration
3. 🔄 **Then**: Implement parallel inference
4. 🔄 **Finally**: Add plate-vehicle association and speed estimation

## Notes

- Both models can run on same Triton server (different model names)
- Use thread pool for parallel CPU preprocessing
- GPU inference happens sequentially but preprocessing is parallel
- Association logic is critical for accuracy
