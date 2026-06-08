from __future__ import annotations

import json
import time

import structlog
from openai import AsyncOpenAI

from inference.config import settings
from inference.schemas import TASK_SCHEMAS

logger = structlog.get_logger()

_client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="not-needed")


async def infer(task: str, messages: list[dict]) -> tuple[dict, int, int, int, int, float | None]:
    """
    Returns (output_dict, prompt_tokens, completion_tokens, latency_ms, ttft_ms, itl_ms).
    Uses streaming to capture TTFT and per-chunk timestamps for ITL; guided_json enforces structured output.
    itl_ms is the avg gap between consecutive token-bearing chunks (excludes TTFT).
    """
    schema_cls = TASK_SCHEMAS.get(task)
    schema = schema_cls.model_json_schema() if schema_cls else None

    extra_body: dict = {}
    if schema:
        extra_body["guided_json"] = schema

    t0 = time.monotonic()
    ttft_ms = 0
    first_token_seen = False
    token_times: list[float] = []
    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    stream = await _client.chat.completions.create(
        model=settings.model_name,
        messages=messages,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body if extra_body else None,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.monotonic()
            if not first_token_seen:
                ttft_ms = int((now - t0) * 1000)
                first_token_seen = True
            token_times.append(now)
            chunks.append(chunk.choices[0].delta.content)
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    latency_ms = int((time.monotonic() - t0) * 1000)
    content = "".join(chunks)

    # avg time between consecutive token-bearing chunks; None if fewer than 2 chunks
    itl_ms: float | None = None
    if len(token_times) > 1:
        itl_ms = round((token_times[-1] - token_times[0]) / (len(token_times) - 1) * 1000, 2)

    try:
        output = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("json_parse_failed", task=task, content=content[:200])
        output = {"raw": content}

    return output, prompt_tokens, completion_tokens, latency_ms, ttft_ms, itl_ms
