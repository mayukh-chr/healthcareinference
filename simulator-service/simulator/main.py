from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import BackgroundTasks, FastAPI, Query
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from simulator.config import settings
from simulator.orchestrator import generate_encounter_batch
from simulator.streaming.rabbitmq_publisher import publisher

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await publisher.connect()
    yield
    await publisher.close()


app = FastAPI(
    title="Hospital Simulator",
    description="Synthetic hospital data generator — ground-truth-first architecture",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


class SimulateResponse(BaseModel):
    run_id: str
    n_encounters: int
    seed: int
    status: str


@app.post("/v1/simulate", response_model=SimulateResponse)
async def simulate(
    background_tasks: BackgroundTasks,
    n: int = Query(default=100, ge=1, le=50_000, description="Number of encounters to generate"),
    seed: int = Query(default=42, description="Global random seed for deterministic generation"),
) -> SimulateResponse:
    run_id = str(uuid.uuid4())
    background_tasks.add_task(generate_encounter_batch, run_id=run_id, n=n, global_seed=seed)
    logger.info("simulation_started", run_id=run_id, n=n, seed=seed)
    return SimulateResponse(run_id=run_id, n_encounters=n, seed=seed, status="running")


@app.get("/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name}
