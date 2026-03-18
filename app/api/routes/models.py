import json
import logging
from pathlib import Path

from fastapi import APIRouter

from app.config import settings
from app.schemas.response import ModelMetrics, ModelsInfoResponse

logger = logging.getLogger(__name__)

router = APIRouter()

STRATEGIES = ["specialized", "multitask", "transfer"]


@router.get("/models/info", response_model=ModelsInfoResponse)
async def models_info():
    from app.main import model_manager

    metrics = []
    models_dir = Path(settings.models_dir)

    for strategy in STRATEGIES:
        metrics_file = models_dir / strategy / "metrics.json"
        if metrics_file.exists():
            try:
                data = json.loads(metrics_file.read_text())
                metrics.append(
                    ModelMetrics(
                        strategy=strategy,
                        age_mae=data.get("age_mae"),
                        gender_accuracy=data.get("gender_accuracy"),
                        ethnicity_accuracy=data.get("ethnicity_accuracy"),
                    )
                )
            except Exception as e:
                logger.error("Failed to read metrics for %s: %s", strategy, e)
                metrics.append(ModelMetrics(strategy=strategy))
        else:
            metrics.append(ModelMetrics(strategy=strategy))

    return ModelsInfoResponse(
        strategies=STRATEGIES,
        metrics=metrics,
        demo_mode=model_manager.is_demo_mode,
    )
