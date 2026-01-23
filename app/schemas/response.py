from pydantic import BaseModel


class FaceBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class AttributePrediction(BaseModel):
    label: str
    value: str
    confidence: float


class PredictionResponse(BaseModel):
    success: bool = True
    strategy: str
    face_detected: bool
    face_box: FaceBox | None = None
    predictions: list[AttributePrediction] = []


class ErrorResponse(BaseModel):
    detail: str
