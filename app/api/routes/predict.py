import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.core.preprocessing import load_image_from_bytes, validate_image
from app.schemas.response import (
    AttributePrediction,
    FaceBox,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def _run_prediction(file: UploadFile, strategy: str) -> PredictionResponse:
    """Pipeline commun : upload -> validation -> détection -> prédiction."""
    from app.main import face_detector, model_manager

    # Lire et valider l'image
    image_bytes = await file.read()
    try:
        validate_image(image_bytes, max_size_mb=settings.max_image_size_mb)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))

    # Charger l'image PIL
    try:
        image = load_image_from_bytes(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Détecter le visage
    detection = face_detector.detect(image)
    if detection is None:
        return PredictionResponse(
            success=True,
            strategy=strategy,
            face_detected=False,
            predictions=[],
        )

    # Prédiction selon la stratégie
    face_tensor = detection["face_tensor"]
    if strategy == "specialized":
        preds = model_manager.predict_specialized(face_tensor)
    elif strategy == "multitask":
        preds = model_manager.predict_multitask(face_tensor)
    elif strategy == "transfer":
        preds = model_manager.predict_transfer(face_tensor)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

    return PredictionResponse(
        success=True,
        strategy=strategy,
        face_detected=True,
        face_box=FaceBox(
            x1=detection["box"]["x1"],
            y1=detection["box"]["y1"],
            x2=detection["box"]["x2"],
            y2=detection["box"]["y2"],
            confidence=detection["confidence"],
        ),
        predictions=[AttributePrediction(**p) for p in preds],
    )


@router.post("/specialized", response_model=PredictionResponse)
async def predict_specialized(file: UploadFile = File(...)):
    """Prédiction avec modèles spécialisés (un modèle par attribut)."""
    return await _run_prediction(file, "specialized")


@router.post("/multitask", response_model=PredictionResponse)
async def predict_multitask(file: UploadFile = File(...)):
    """Prédiction avec modèle multitâche (un seul modèle, plusieurs sorties)."""
    return await _run_prediction(file, "multitask")


@router.post("/transfer", response_model=PredictionResponse)
async def predict_transfer(file: UploadFile = File(...)):
    """Prédiction avec modèle transfer learning."""
    return await _run_prediction(file, "transfer")
