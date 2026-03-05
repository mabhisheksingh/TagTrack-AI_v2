from pydantic import BaseModel, Field
from typing import List, Optional


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    ocr_text: Optional[str] = None


class DetectionWithTracking(Detection):
    tracker_id: int


class ProcessImageResponse(BaseModel):
    detections: List[DetectionWithTracking]
    message: str = "Image processed successfully"


class ProcessBatchRequest(BaseModel):
    image_paths: List[str] = Field(..., description="List of image file paths to process")


class ProcessBatchResponse(BaseModel):
    processed: int
    results: List[dict]


class ProcessFolderRequest(BaseModel):
    input_folder: str = Field(..., description="Path to input folder containing images")
    output_folder: str = Field(..., description="Path to output folder for annotated images")


class ProcessFolderResponse(BaseModel):
    processed: int
    results: List[dict]
    message: str = "Folder processed successfully"
