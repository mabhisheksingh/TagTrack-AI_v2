from fastapi import APIRouter
from datetime import datetime, timezone
from app.core.config import settings

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "triton_server": settings.triton_server_url,
        "model": settings.triton_model_name,
        "time": datetime.now(timezone.utc).isoformat(),
    }
