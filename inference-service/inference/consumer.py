from __future__ import annotations

import json
import uuid
from collections import defaultdict

import aio_pika
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from inference.config import settings
from inference.prompt_builder import build_prompt
from inference.vllm_client import infer

logger = structlog.get_logger()

engine = create_async_engine(settings.database_url)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# In-memory buffer: encounter_id → list of document content strings
_encounter_buffer: dict[str, list[str]] = defaultdict(list)
_TASKS = ["diagnosis", "entity_extraction", "structured_json", "summarization", "risk_classification"]


async def start_consumer() -> None:
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=10)

    # Consume all document queues
    for queue_name in ("admissions", "labs", "clinical_notes", "discharge"):
        queue = await channel.declare_queue(queue_name, durable=True)
        await queue.consume(_handle_message)

    logger.info("rabbitmq_consumer_started")


async def _handle_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    async with message.process():
        body = json.loads(message.body.decode())
        routing_key = message.routing_key or ""
        encounter_id = body.get("encounter_id", "")

        if not encounter_id:
            return

        # Buffer document content
        content = body.get("content") or body.get("payload", {})
        if isinstance(content, str) and content:
            _encounter_buffer[encounter_id].append(content)
        elif isinstance(content, dict):
            _encounter_buffer[encounter_id].append(json.dumps(content, indent=2))

        # Trigger inference on discharge
        if routing_key == "hospital.events.discharge":
            docs = _encounter_buffer.pop(encounter_id, [])
            if docs:
                await _run_inference_pipeline(encounter_id, docs)


async def _run_inference_pipeline(encounter_id: str, documents: list[str]) -> None:
    logger.info("inference_started", encounter_id=encounter_id, n_docs=len(documents))
    async with SessionLocal() as session:
        for task in _TASKS:
            messages = build_prompt(task, documents)
            try:
                output, prompt_tokens, completion_tokens, latency_ms, ttft_ms = await infer(task, messages)
                await _store_result(
                    session, encounter_id, task, output, prompt_tokens, completion_tokens, latency_ms, ttft_ms
                )
            except Exception as exc:
                logger.error("inference_task_failed", task=task, encounter_id=encounter_id, error=str(exc))
        await session.commit()
    logger.info("inference_completed", encounter_id=encounter_id)


async def _store_result(
    session: AsyncSession,
    encounter_id: str,
    task: str,
    output: dict,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    ttft_ms: int,
) -> None:
    from sqlalchemy import text
    await session.execute(
        text("""
            INSERT INTO inference_results
                (result_id, encounter_id, model_name, task, inferred_output,
                 prompt_tokens, completion_tokens, latency_ms, ttft_ms)
            VALUES
                (:result_id, :encounter_id, :model_name, :task, :inferred_output::jsonb,
                 :prompt_tokens, :completion_tokens, :latency_ms, :ttft_ms)
        """),
        {
            "result_id": str(uuid.uuid4()),
            "encounter_id": encounter_id,
            "model_name": settings.model_name,
            "task": task,
            "inferred_output": json.dumps(output),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
        },
    )
