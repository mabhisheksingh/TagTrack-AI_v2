from typing import Annotated, List
import time
import structlog
import shutil
import os
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
import tempfile
import uuid

from app.services.anpr_service import ANPRService, get_anpr_service
from app.core.config import settings

router = APIRouter(prefix="/v1/anpr")
logger = structlog.get_logger(__name__)


ServiceDep = Annotated[ANPRService, Depends(get_anpr_service)]

@router.post("/process-image-video" )
async def process_image_video(files: List[UploadFile] = File(..., description="List of files to process")
                    , service: ServiceDep = None
                    , request_id: str = Header(None, description="Request ID")):
    logger.info("api.process_image_video", request_id=request_id)
    api_start = time.perf_counter()
    temp_dir = None
    if request_id is None:
        request_id = str(uuid.uuid4())
    try:
        temp_dir = tempfile.mkdtemp(prefix="anpr_api_"+request_id)
        temp_input = os.path.join(temp_dir, "input")
        os.makedirs(temp_input)
        unsupported_list = []
        supported_list =[]

        for file in files:
            # Extract the extension (e.g., '.mp4' or '.jpg')
            _, file_extension = os.path.splitext(file.filename)
            if file_extension not in settings.combined_extensions_list:
                unsupported_list.append(file.filename)
            else:
                supported_list.append(file.filename)
                temp_file = os.path.join(temp_input, file.filename)
                with open(temp_file, 'wb') as f:
                    shutil.copyfileobj(file.file, f)
        
        await service.process_videos_and_images(input_folder=temp_input,request_id=request_id)

        if len(unsupported_list) > 0:
            raise HTTPException(status_code=400, detail=f"Unsupported file types: {unsupported_list}")

        return JSONResponse(content={
            "message": "Files processed successfully",
            "supported_list": supported_list,
            "unsupported_list": unsupported_list},
            status_code=200
        )
    except Exception as e:
        logger.exception("api.process_audio_video.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        total_duration = int((time.perf_counter() - api_start) * 1000)
        logger.info("api.process_audio_video", total_duration=total_duration)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# @router.post("/process-image", response_model=ProcessImageResponse)
# async def process_image(file: UploadFile = File(...), service: ServiceDep = None) -> ProcessImageResponse:
#     api_start = time.perf_counter()
#
#     temp_dir = None
#     try:
#         temp_dir = tempfile.mkdtemp(prefix="anpr_api_")
#         temp_input = Path(temp_dir) / "input"
#         temp_output = Path(temp_dir) / "output"
#         temp_input.mkdir()
#         temp_output.mkdir()
#
#         temp_file = temp_input / file.filename
#         contents = await file.read()
#         with open(temp_file, 'wb') as f:
#             f.write(contents)
#
#         image = cv2.imread(str(temp_file))
#         if image is None:
#             raise HTTPException(status_code=400, detail="Invalid image file")
#
#         annotated_frame, results = await service.process_frame(image)
#
#         output_file = temp_output / file.filename
#         cv2.imwrite(str(output_file), annotated_frame)
#
#         final_output_dir = settings.data_output_dir
#         final_output_dir.mkdir(parents=True, exist_ok=True)
#         final_output_path = final_output_dir / file.filename
#         shutil.copy2(output_file, final_output_path)
#
#         detections = [DetectionWithTracking(**result) for result in results]
#
#         total_ms = int((time.perf_counter() - api_start) * 1000)
#         logger.info("api.process_image",
#                    detections_count=len(detections),
#                    output_path=str(final_output_path),
#                    total_duration_ms=total_ms)
#
#         return ProcessImageResponse(detections=detections)
#
#     except Exception as e:
#         logger.exception("api.process_image.error", error=str(e))
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         end_time = time.perf_counter()
#         total_time = end_time - api_start
#         logger.info("api.process_image", total_time=total_time)
#         if temp_dir and os.path.exists(temp_dir):
#             shutil.rmtree(temp_dir)


# @router.post("/process-batch", response_model=ProcessBatchResponse)
# async def process_batch(request: ProcessBatchRequest, service: ServiceDep) -> ProcessBatchResponse:
#     api_start = time.perf_counter()
#
#     temp_dir = None
#     try:
#         temp_dir = tempfile.mkdtemp(prefix="anpr_batch_")
#         temp_input = Path(temp_dir) / "input"
#         temp_output = Path(temp_dir) / "output"
#         temp_input.mkdir()
#         temp_output.mkdir()
#
#         temp_paths = []
#         for img_path in request.image_paths:
#             src = Path(img_path)
#             if src.exists():
#                 dst = temp_input / src.name
#                 shutil.copy2(src, dst)
#                 temp_paths.append(str(dst))
#
#         # process_batch is still synchronous in anpr_service, but we'll call it from async route
#         # If you want to make it async, we'd need to update anpr_service.py's process_batch too
#         results = service.process_batch(temp_paths)
#
#         final_output_dir = settings.data_output_dir
#         final_output_dir.mkdir(parents=True, exist_ok=True)
#
#         for result in results:
#             if "output_path" in result:
#                 output_file = Path(result["output_path"])
#                 if output_file.exists():
#                     final_path = final_output_dir / output_file.name
#                     shutil.copy2(output_file, final_path)
#                     result["final_output_path"] = str(final_path)
#
#         total_ms = int((time.perf_counter() - api_start) * 1000)
#         logger.info("api.process_batch",
#                    batch_size=len(request.image_paths),
#                    processed=len(results),
#                    total_duration_ms=total_ms)
#
#         return ProcessBatchResponse(processed=len(results), results=results)
#
#     except Exception as e:
#         logger.error("api.process_batch.error", error=str(e))
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if temp_dir and os.path.exists(temp_dir):
#             shutil.rmtree(temp_dir)
#
#
# @router.post("/process-folder", response_model=ProcessFolderResponse)
# async def process_folder(request: ProcessFolderRequest, service: ServiceDep) -> ProcessFolderResponse:
#     api_start = time.perf_counter()
#
#     temp_dir = None
#     try:
#         temp_dir = tempfile.mkdtemp(prefix="anpr_folder_")
#         temp_input = Path(temp_dir) / "input"
#         temp_output = Path(temp_dir) / "output"
#         temp_input.mkdir()
#         temp_output.mkdir()
#
#         input_folder = Path(request.input_folder)
#         if not input_folder.exists():
#             raise HTTPException(status_code=404, detail=f"Input folder not found: {request.input_folder}")
#
#         image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
#         for img_file in input_folder.iterdir():
#             if img_file.is_file() and img_file.suffix.lower() in image_extensions:
#                 shutil.copy2(img_file, temp_input / img_file.name)
#
#         # process_folder is still synchronous in anpr_service
#         result = service.process_folder(str(temp_input), str(temp_output))
#
#         final_output_dir = Path(request.output_folder)
#         final_output_dir.mkdir(parents=True, exist_ok=True)
#
#         for output_file in temp_output.iterdir():
#             if output_file.is_file():
#                 final_path = final_output_dir / output_file.name
#                 shutil.copy2(output_file, final_path)
#
#         total_ms = int((time.perf_counter() - api_start) * 1000)
#         logger.info("api.process_folder",
#                    processed=result["processed"],
#                    output_folder=request.output_folder,
#                    total_duration_ms=total_ms)
#
#         return ProcessFolderResponse(
#             processed=result["processed"],
#             results=result["results"],
#             message=f"Folder processed successfully. Output saved to {request.output_folder}"
#         )
#
#     except Exception as e:
#         logger.error("api.process_folder.error", error=str(e))
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if temp_dir and os.path.exists(temp_dir):
#             shutil.rmtree(temp_dir)
