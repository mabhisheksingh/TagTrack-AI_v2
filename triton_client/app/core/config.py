from typing import List
from pathlib import Path
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Centralized configuration management with validation and robust error handling."""
    
    # ============================================================================
    # APPLICATION SETTINGS
    # ============================================================================
    app_name: str = Field("ANPR API", validation_alias="APP_NAME")
    app_env: str = Field("development", validation_alias="APP_ENV")
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")
    
    # ============================================================================
    # TRITON SERVER CONFIGURATION
    # ============================================================================
    triton_server_url: str = Field("localhost:8001", validation_alias="TRITON_SERVER_URL", 
                                   description="Triton server URL (gRPC port, typically 8001)")
    triton_model_name: str = Field(..., validation_alias="TRITON_MODEL_NAME")
    triton_protocol: str = Field("grpc", validation_alias="TRITON_PROTOCOL", 
                                description="Protocol: 'grpc' (recommended) or 'http'")
    
    # Model class configuration (comma-separated class names in order)
    model_class_names: str = Field("number_plate", validation_alias="MODEL_CLASS_NAMES",
                                   description="Comma-separated class names for the model (e.g., 'number_plate' or 'vehicle,car,truck,bus')")
    ocr_class_ids: str = Field("0", validation_alias="OCR_CLASS_IDS",
                              description="Comma-separated class IDs that should trigger OCR (e.g., '0' or '1,2')")
    
    # ============================================================================
    # PROCESSING CONFIGURATION
    # ============================================================================
    batch_size: int = Field(4, validation_alias="BATCH_SIZE", ge=1, le=128)
    input_folder: str = Field("data/input", validation_alias="INPUT_FOLDER")
    output_folder: str = Field("data/output", validation_alias="OUTPUT_FOLDER")
    models_dir: str = Field("models", validation_alias="MODELS_DIR")
    
    # ============================================================================
    # DETECTION SETTINGS
    # ============================================================================
    confidence_threshold: float = Field(0.5, validation_alias="CONFIDENCE_THRESHOLD", ge=0.0, le=1.0)
    ocr_trigger_confidence_threshold: float = Field(0.7, validation_alias="OCR_TRIGGER_CONFIDENCE_THRESHOLD", ge=0.0, le=1.0)
    iou_threshold: float = Field(0.5, validation_alias="IOU_THRESHOLD", ge=0.0, le=1.0)
    blur_threshold: float = Field(40.0, validation_alias="BLUR_THRESHOLD", ge=0.0)
    plate_blur_threshold: float = Field(
        60.0,
        validation_alias="PLATE_BLUR_THRESHOLD",
        ge=0.0,
        description="Minimum Laplacian variance required on the cropped plate before triggering OCR",
    )
    
    # ============================================================================
    # VIDEO PROCESSING
    # ============================================================================
    enable_video_detection: bool = Field(True, validation_alias="ENABLE_VIDEO_DETECTION")
    frames_per_second: float = Field(1.0, validation_alias="FRAMES_PER_SECOND", ge=0.1, le=30.0)
    
    # ============================================================================
    # FILE EXTENSIONS
    # ============================================================================
    video_extensions: str = Field(".mp4,.avi,.mov,.mkv,.webm", validation_alias="VIDEO_EXTENSIONS")
    image_extensions: str = Field(".jpg,.jpeg,.png,.bmp,.JPG,.JPEG,.PNG", validation_alias="IMAGE_EXTENSIONS")
    
    # ============================================================================
    # OCR CONFIGURATION
    # ============================================================================
    enable_ocr: bool = Field(True, validation_alias="ENABLE_OCR", 
                            description="Enable/disable OCR processing (set to False if OCR hangs)")
    ocr_timeout: int = Field(3, validation_alias="OCR_TIMEOUT", ge=1, le=30,
                            description="OCR timeout in seconds")
    ocr_languages: str = Field("en", validation_alias="OCR_LANGUAGES")
    x_padding: float = Field(0.20, validation_alias="X_PADDING", ge=0.0, le=1.0)
    y_padding: float = Field(0.20, validation_alias="Y_PADDING", ge=0.0, le=1.0)
    
    # InternVL2 OCR (Optional)
    use_internvl2: bool = Field(False, validation_alias="USE_INTERNVL2")
    internvl2_model_path: str = Field("OpenGVLab/InternVL2-1B", validation_alias="INTERNVL2_MODEL_PATH")
    internvl2_device: str = Field("cuda", validation_alias="INTERNVL2_DEVICE")
    internvl2_input_size: int = Field(448, validation_alias="INTERNVL2_INPUT_SIZE")
    
    # ============================================================================
    # CEPH/S3 STORAGE (Optional)
    # ============================================================================
    ceph_access_key: str = Field("", validation_alias="CEPH_ACCESS_KEY")
    ceph_secret_access_key: str = Field("", validation_alias="CEPH_SECRET_ACCESS_KEY")
    ceph_region: str = Field("us-east-1", validation_alias="CEPH_REGION")
    ceph_bucket_name: str = Field("", validation_alias="CEPH_BUCKET_NAME")
    ceph_endpoint_url: str = Field("", validation_alias="CEPH_ENDPOINT_URL")
    
    model_config = SettingsConfigDict(
        # Resolve .env relative to project root so running from repo root loads it correctly
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )
    
    # ============================================================================
    # COMPUTED PROPERTIES
    # ============================================================================
    
    @computed_field
    @property
    def video_extensions_list(self) -> List[str]:
        """Parse video extensions into a list."""
        return [ext.strip() for ext in self.video_extensions.split(',') if ext.strip()]
    
    @computed_field
    @property
    def image_extensions_list(self) -> List[str]:
        """Parse image extensions into a list."""
        return [ext.strip() for ext in self.image_extensions.split(',') if ext.strip()]

    @computed_field
    @property
    def combined_extensions_list(self) -> List[str]:
        """Parse extensions into a list."""
        extensions = (
            self.video_extensions.split(',') +
            self.image_extensions.split(',')
        )
        return [ext.strip() for ext in extensions if ext.strip()]
    
    @computed_field
    @property
    def class_names_list(self) -> List[str]:
        """Parse model class names into a list."""
        return [name.strip() for name in self.model_class_names.split(',') if name.strip()]
    
    @computed_field
    @property
    def ocr_class_ids_list(self) -> List[int]:
        """Parse OCR class IDs into a list of integers."""
        return [int(id.strip()) for id in self.ocr_class_ids.split(',') if id.strip().isdigit()]
    
    @computed_field
    @property
    def data_input_dir(self) -> Path:
        """Get absolute path to input directory."""
        return (PROJECT_ROOT / self.input_folder).resolve()
    
    @computed_field
    @property
    def data_output_dir(self) -> Path:
        """Get absolute path to output directory."""
        return (PROJECT_ROOT / self.output_folder).resolve()
    
    @computed_field
    @property
    def models_path(self) -> Path:
        """Get absolute path to models directory."""
        return (PROJECT_ROOT / self.models_dir).resolve()
    
    # ============================================================================
    # CONSTANTS
    # ============================================================================
    
    DETECT_CLASSES: List[int] = [0]  # YOLO class IDs to keep
    CSV_FRAME_HEADER: List[str] = ["frame", "track_id", "plate_text", "confidence", "confidence_ocr"]
    CSV_TRACK_SUMMARY_HEADER: List[str] = [
        "track_id",
        "plate_text",
        "votes",
        "avg_confidence",
        "avg_confidence_ocr",
    ]
    CSV_CONSOLIDATED_HEADER: List[str] = [
        "source",
        "track_id",
        "plate_text",
        "votes",
        "avg_confidence",
        "avg_confidence_ocr",
    ]
    LABELS_SUBDIR: str = "labels"
    OCR_RESULT_SUBDIR: str = "ocr_result"
    
    # ============================================================================
    # VALIDATION & UTILITY METHODS
    # ============================================================================
    
    def create_directories(self) -> None:
        """Create all required directories with error handling."""
        directories = [
            self.data_input_dir,
            self.data_output_dir,
            self.models_path,
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ensured: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {str(e)}")
                raise
    
    def validate_config(self) -> None:
        """Validate critical configuration values."""
        errors = []
        
        if not self.triton_server_url:
            errors.append("TRITON_SERVER_URL is required")
        
        if not self.triton_model_name:
            errors.append("TRITON_MODEL_NAME is required")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def get_summary(self) -> dict:
        """Get configuration summary for logging."""
        return {
            "app_name": self.app_name,
            "app_env": self.app_env,
            "triton_server": self.triton_server_url,
            "triton_model": self.triton_model_name,
            "batch_size": self.batch_size,
            "confidence_threshold": self.confidence_threshold,
            "video_enabled": self.enable_video_detection,
            "fps": self.frames_per_second,
            "input_folder": str(self.data_input_dir),
            "output_folder": str(self.data_output_dir),
        }


# Initialize settings instance
settings = Settings()

# Create directories and validate on module import
try:
    settings.create_directories()
    settings.validate_config()
    logger.info("Settings initialized successfully", config=settings.get_summary())
except Exception as e:
    logger.error(f"Failed to initialize settings: {str(e)}")
    raise
