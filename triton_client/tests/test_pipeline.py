import numpy as np
import pytest
from unittest.mock import MagicMock

from src.pipeline import ProcessingPipeline

@pytest.fixture
def mock_triton_client():
    """Fixture for a mock TritonClient."""
    client = MagicMock()
    # Mock the infer method to return a sample detection
    # The format should be similar to what a YOLO model would output
    # [x_center, y_center, width, height, confidence, class_id]
    sample_detection = np.array([
        [0.5, 0.5, 0.2, 0.2, 0.9, 0],  # A vehicle
        [0.6, 0.6, 0.1, 0.05, 0.95, 1] # A number plate
    ])
    client.infer.return_value = sample_detection
    return client

@pytest.fixture
def mock_ocr():
    """Fixture for a mock Ocr."""
    ocr = MagicMock()
    ocr.recognize_text.return_value = "TEST-123"
    return ocr

def test_processing_pipeline(mock_triton_client, mock_ocr):
    """Test the ProcessingPipeline with mock dependencies."""
    # Patch the Ocr class in the pipeline module
    with pytest.MonkeyPatch.context() as m:
        m.setattr("app.pipeline.Ocr", lambda: mock_ocr)

        # Initialize the pipeline with the mock Triton client
        pipeline = ProcessingPipeline(triton_client=mock_triton_client)

        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Process the frame
        annotated_frame = pipeline.process_frame(dummy_image)

        # Assert that the output is not empty
        assert annotated_frame is not None
        assert annotated_frame.shape == dummy_image.shape

        # Check if the infer method was called
        mock_triton_client.infer.assert_called_once()

        # Check if OCR was called for the number plate
        mock_ocr.recognize_text.assert_called_once()
