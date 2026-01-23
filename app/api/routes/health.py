from fastapi import APIRouter

from app.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    from app.main import model_manager

    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "models_loaded": model_manager.loaded_count,
        "demo_mode": model_manager.is_demo_mode,
    }
