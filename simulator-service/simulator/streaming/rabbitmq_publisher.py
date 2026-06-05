from __future__ import annotations

import json
from datetime import datetime

import aio_pika
import structlog

from simulator.config import settings
from simulator.models.documents import ClinicalDocument
from simulator.models.events import ClinicalEvent
from simulator.models.ground_truth import GroundTruthLabels
from simulator.models.labs import LabResult

logger = structlog.get_logger()


class HospitalEventPublisher:
    def __init__(self) -> None:
        self._connection: aio_pika.abc.AbstractConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        self._channel = await self._connection.channel()
        self._exchange = await self._channel.declare_exchange(
            settings.rabbitmq_exchange,
            aio_pika.ExchangeType.TOPIC,
            durable=True,
        )
        await self._declare_queues()
        logger.info("rabbitmq_connected", exchange=settings.rabbitmq_exchange)

    async def _declare_queues(self) -> None:
        assert self._channel is not None
        queue_bindings = {
            "admissions": "hospital.events.admission",
            "labs": "hospital.events.lab",
            "clinical_notes": "hospital.events.note",
            "treatment_updates": "hospital.events.treatment",
            "discharge": "hospital.events.discharge",
            "ground_truth": "hospital.events.ground_truth",
        }
        for queue_name, routing_key in queue_bindings.items():
            queue = await self._channel.declare_queue(queue_name, durable=True)
            assert self._exchange is not None
            await queue.bind(self._exchange, routing_key=routing_key)

    async def publish_event(self, event: ClinicalEvent) -> None:
        assert self._exchange is not None
        body = event.model_dump_json().encode()
        msg = aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        await self._exchange.publish(msg, routing_key=event.routing_key())

    async def publish_document(self, doc: ClinicalDocument) -> None:
        assert self._exchange is not None
        body = doc.model_dump_json().encode()
        msg = aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        await self._exchange.publish(msg, routing_key="hospital.events.note")

    async def publish_ground_truth(self, labels: GroundTruthLabels) -> None:
        assert self._exchange is not None
        body = labels.model_dump_json().encode()
        msg = aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        await self._exchange.publish(msg, routing_key="hospital.events.ground_truth")

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()


# Module-level singleton; initialized at app startup
publisher = HospitalEventPublisher()
