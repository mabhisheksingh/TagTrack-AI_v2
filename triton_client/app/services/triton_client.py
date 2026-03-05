import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype
from typing import Dict, Tuple, Any
import cv2
import structlog
import traceback


class TritonClient:
    def __init__(self, server_url: str, model_name: str):
        self.logger = structlog.get_logger(__name__)
        self.model_name = model_name
        self.server_url = server_url.replace('http://', '').replace('https://', '')
        self.triton_client = None

    async def _get_client(self):
        if self.triton_client is None:
            try:
                self.triton_client = grpcclient.InferenceServerClient(url=self.server_url)
                if await self.triton_client.is_server_live():
                    self.logger.info("triton_connected", server=self.server_url, protocol="gRPC (async)")
                else:
                    raise ConnectionError(f"Triton server at {self.server_url} is not live")
            except Exception as e:
                self.logger.error("triton_connection_failed", error=str(e))
                raise
        return self.triton_client

    def preprocess_image(self, image: np.ndarray, frame_idx: int, target_size: Tuple[int, int] = (640, 640)) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        # ... (Your exact preprocessing code here, it is perfectly correct) ...
        original_height, original_width = image.shape[:2]
        target_height, target_width = target_size
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized = cv2.resize(image, (new_width, new_height))
        padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        meta = {"scale": scale, "x_offset": x_offset, "y_offset": y_offset, "input_size": target_size,
                "frame_idx": frame_idx}
        return blob, meta

    # Notice we just await the client directly here. No self.tasks needed!
    async def infer(self, image: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            client = await self._get_client()
            preprocessed, preprocess_meta = self.preprocess_image(image, frame_idx)
            preprocessed = np.expand_dims(preprocessed, axis=0)

            inputs = [grpcclient.InferInput("images", preprocessed.shape, np_to_triton_dtype(preprocessed.dtype))]
            inputs[0].set_data_from_numpy(preprocessed)
            outputs = [grpcclient.InferRequestedOutput("output0")]

            # Send the async request to Triton
            result = await client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)

            # Unpack and return
            detections = result.as_numpy("output0")
            return detections, preprocess_meta

        except Exception as e:
            self.logger.error("inference_error", error=str(e), traceback=traceback.format_exc())
            raise

    async def close(self):
        if self.triton_client is not None:
            await self.triton_client.close()
            self.triton_client = None