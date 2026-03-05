import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from app.services.anpr_service import ANPRService
from app.services.triton_client import TritonClient
from app.services.ocr import OCRService


@pytest.fixture
def mock_triton_client():
    client = Mock(spec=TritonClient)
    client.infer.return_value = np.random.rand(1, 6, 8400)
    return client


@pytest.fixture
def mock_ocr_service():
    service = Mock(spec=OCRService)
    service.recognize_text.return_value = "ABC123"
    return service


@pytest.fixture
def anpr_service(mock_triton_client, mock_ocr_service):
    return ANPRService(triton_client=mock_triton_client, ocr_service=mock_ocr_service)


def test_anpr_service_initialization(anpr_service):
    assert anpr_service is not None
    assert anpr_service.tracker is not None
    assert anpr_service.box_annotator is not None
    assert anpr_service.label_annotator is not None


def test_process_frame(anpr_service):
    frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    annotated_frame, results = anpr_service.process_frame(frame)
    
    assert annotated_frame is not None
    assert isinstance(results, list)
    assert annotated_frame.shape == frame.shape
