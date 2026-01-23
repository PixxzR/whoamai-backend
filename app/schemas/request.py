from pydantic import BaseModel, Field


class PredictionOptions(BaseModel):
    """Options optionnelles pour la prédiction."""

    return_face_box: bool = Field(
        default=True, description="Inclure les coordonnées du visage"
    )
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Seuil de confiance minimum"
    )
