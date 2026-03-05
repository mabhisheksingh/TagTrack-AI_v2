"""
OCR Engine implementations using PaddleOCR.
"""

import cv2
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from paddleocr import PaddleOCR
import structlog
logger = structlog.get_logger(__name__)

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
class PaddleOCREngine:
    """PaddleOCR wrapper for license plate text recognition."""

    def __init__(
        self,
        use_angle_cls: bool = True,
        lang: str = 'en',
        use_gpu: bool = False,
        show_log: bool = False,
    ):
        """
        Initialize PaddleOCR engine.

        Args:
            use_angle_cls: Whether to use angle classification
            lang: Language code for OCR
            use_gpu: Whether to use GPU acceleration
            show_log: Whether to show PaddleOCR logs
        """
        logger.debug(f"Initializing PaddleOCR engine, lang={lang}, use_gpu={use_gpu}")
        self.reader = PaddleOCR(
            use_textline_orientation=True,
            lang=lang
        )
        logger.debug("PaddleOCR engine initialized")

    def recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Perform OCR on an image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            List of dicts with 'text' and 'confidence' keys
        """
        # Ensure image is in BGR format for PaddleOCR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # paddleocr 3.x API
        ocr_result = self.reader.predict(image)

        results = []
        for res in ocr_result:
            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])
            if texts and scores:
                logger.info(f"OCR result: {texts} with confidence {scores}")
                results.append({
                    'text':  "".join(texts).strip(),
                    'confidence': float(scores[0])
                })

        logger.debug(f"OCR completed, result_count={len(results)}")
        return results

    def recognize_text_only(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Perform OCR and return concatenated text with average confidence.

        Args:
            image: Input image

        Returns:
            Tuple of (concatenated_text, average_confidence)
        """
        results = self.recognize(image)
        if not results:
            return "", 0.0

        all_text = "".join(r['text'] for r in results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)

        return all_text, avg_confidence
