from pydantic import BaseModel, Field


class FaceBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float


class PredictionData(BaseModel):
    age: float
    gender: str
    gender_confidence: float
    ethnicity: str
    ethnicity_confidence: float
    ethnicity_class_id: int


class PredictionMetadata(BaseModel):
    strategy: str
    model_version: str = "1.0.0"
    inference_time_ms: float
    demo_mode: bool
    face_detection: FaceBox | None = None


class PredictionResponse(BaseModel):
    success: bool = True
    data: PredictionData
    metadata: PredictionMetadata


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict | None = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: int
    demo_mode: bool
    uptime: float = Field(description="Uptime in seconds")


class ModelMetrics(BaseModel):
    strategy: str
    age_mae: float | None = None
    gender_accuracy: float | None = None
    ethnicity_accuracy: float | None = None


class ModelsInfoResponse(BaseModel):
    strategies: list[str]
    current_default: str = "multitask"
    metrics: list[ModelMetrics] = []
    demo_mode: bool
