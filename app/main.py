import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health, predict
from app.config import settings
from app.core.detection import FaceDetector
from app.core.inference import ModelManager

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Instances globales
face_detector = FaceDetector(device=settings.device)
model_manager = ModelManager(models_dir=settings.models_dir, device=settings.device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Device: %s", settings.device)
    face_detector.load()
    model_manager.load_all()
    if model_manager.is_demo_mode:
        logger.info("DEMO MODE: predictions are random (no .pth models found)")
    yield
    # Shutdown
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug("%s %s", request.method, request.url.path)
    response = await call_next(request)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
