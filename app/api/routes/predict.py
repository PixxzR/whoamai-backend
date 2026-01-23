from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.response import PredictionResponse

router = APIRouter()


@router.post("/specialized", response_model=PredictionResponse)
async def predict_specialized(file: UploadFile = File(...)):
    """Prédiction avec modèles spécialisés (un modèle par attribut)."""
    # TODO: implémenter pipeline specialized
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/multitask", response_model=PredictionResponse)
async def predict_multitask(file: UploadFile = File(...)):
    """Prédiction avec modèle multitâche (un seul modèle, plusieurs sorties)."""
    # TODO: implémenter pipeline multitask
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/transfer", response_model=PredictionResponse)
async def predict_transfer(file: UploadFile = File(...)):
    """Prédiction avec modèle transfer learning."""
    # TODO: implémenter pipeline transfer
    raise HTTPException(status_code=501, detail="Not implemented yet")
