from __future__ import annotations

import asyncio

import structlog
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from inference.config import settings
from inference.consumer import start_consumer

logger = structlog.get_logger()

app = FastAPI(
    title="Hospital Inference Service",
    description="RabbitMQ consumer + vLLM inference for clinical benchmarking",
    version="0.1.0",
)

Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(start_consumer())
    logger.info("inference_service_started", model=settings.model_name)


@app.get("/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name, "model": settings.model_name}
