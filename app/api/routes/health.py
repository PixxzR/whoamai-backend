import time

from fastapi import APIRouter

from app.config import settings
from app.schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    from app.main import app_state, model_manager

    uptime = time.time() - app_state["startup_time"]

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        models_loaded=model_manager.loaded_count,
        demo_mode=model_manager.is_demo_mode,
        uptime=round(uptime, 2),
    )
