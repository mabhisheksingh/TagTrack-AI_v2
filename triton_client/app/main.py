import uuid

from fastapi import FastAPI
from contextlib import asynccontextmanager

from requests import Request

from app.core.config import settings
from app.core.logging import setup_logger
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.anpr import router as anpr_router


tags_metadata = [
    {"name": "Health", "description": "Service health checks"},
    {"name": "ANPR", "description": "Automatic Number Plate Recognition endpoints"},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    yield

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="API for Automatic Number Plate Recognition using Triton Inference Server and Supervision",
    openapi_tags=tags_metadata,
    contact={
        "name": "ANPR Team",
        "url": "https://example.com",
        "email": "support@example.com",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)
app.include_router(health_router, tags=["Health"])
app.include_router(anpr_router, tags=["ANPR"])


def start_server():
    """Entry point for the anpr-api command."""
    import uvicorn
    setup_logger()
    uvicorn.run("app.main:app", host="0.0.0.0", port=9001, reload=True)


if __name__ == "__main__":
    start_server()
