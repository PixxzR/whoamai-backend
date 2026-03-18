import logging
import time

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.preprocessing import load_image_from_bytes, validate_image
from app.schemas.response import (
    FaceBox,
    PredictionData,
    PredictionMetadata,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _error_response(status_code: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": message,
                "details": details,
            },
        },
    )


async def _run_prediction(file: UploadFile, strategy: str) -> PredictionResponse | JSONResponse:
    """Pipeline commun : upload -> validation -> détection -> prédiction."""
    from app.main import face_detector, model_manager

    # Lire et valider l'image
    try:
        image_bytes = await file.read()
    finally:
        await file.close()

    try:
        validate_image(image_bytes, max_size_mb=settings.max_image_size_mb)
    except ValueError:
        return _error_response(
            413,
            "IMAGE_TOO_LARGE",
            f"Image exceeds maximum size of {settings.max_image_size_mb}MB",
            {"max_size_mb": settings.max_image_size_mb},
        )

    # Charger l'image PIL
    try:
        image = load_image_from_bytes(image_bytes)
    except Exception:
        return _error_response(
            400,
            "INVALID_IMAGE",
            "The uploaded file is not a valid image",
        )

    # Détecter le visage
    detection = face_detector.detect(image)
    if detection is None:
        return _error_response(
            422,
            "NO_FACE_DETECTED",
            "No face was detected in the uploaded image",
        )

    # Prédiction selon la stratégie
    face_tensor = detection["face_tensor"]
    start_time = time.perf_counter()

    if strategy == "specialized":
        preds = model_manager.predict_specialized(face_tensor)
    elif strategy == "multitask":
        preds = model_manager.predict_multitask(face_tensor)
    elif strategy == "transfer":
        preds = model_manager.predict_transfer(face_tensor)
    else:
        return _error_response(400, "INVALID_STRATEGY", f"Unknown strategy: {strategy}")

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    face_box = FaceBox(
        x=detection["box"]["x"],
        y=detection["box"]["y"],
        width=detection["box"]["width"],
        height=detection["box"]["height"],
        confidence=detection["confidence"],
    )

    return PredictionResponse(
        success=True,
        data=PredictionData(**preds),
        metadata=PredictionMetadata(
            strategy=strategy,
            inference_time_ms=round(inference_time_ms, 2),
            demo_mode=model_manager.is_demo_mode,
            face_detection=face_box,
        ),
    )


@router.post("/specialized")
async def predict_specialized(file: UploadFile = File(...)):
    """Prédiction avec modèles spécialisés (un modèle par attribut)."""
    return await _run_prediction(file, "specialized")


@router.post("/multitask")
async def predict_multitask(file: UploadFile = File(...)):
    """Prédiction avec modèle multitâche (un seul modèle, plusieurs sorties)."""
    return await _run_prediction(file, "multitask")


@router.post("/transfer")
async def predict_transfer(file: UploadFile = File(...)):
    """Prédiction avec modèle transfer learning."""
    return await _run_prediction(file, "transfer")
