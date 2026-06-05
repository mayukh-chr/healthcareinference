from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class EncounterORM(Base):
    __tablename__ = "encounters"

    encounter_id = Column(UUID(as_uuid=True), primary_key=True)
    patient_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    encounter_type = Column(String(20), nullable=False)
    primary_disease = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    admitted_at = Column(DateTime(timezone=True), nullable=False)
    discharged_at = Column(DateTime(timezone=True), nullable=True)
    generation_seed = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    events = relationship("EventORM", back_populates="encounter", lazy="dynamic")
    lab_results = relationship("LabResultORM", back_populates="encounter", lazy="dynamic")
    clinical_documents = relationship("ClinicalDocumentORM", back_populates="encounter", lazy="dynamic")
    ground_truth = relationship("GroundTruthORM", back_populates="encounter", uselist=False)


class EventORM(Base):
    __tablename__ = "events"

    event_id = Column(UUID(as_uuid=True), primary_key=True)
    encounter_id = Column(UUID(as_uuid=True), ForeignKey("encounters.encounter_id"), nullable=False)
    patient_id = Column(UUID(as_uuid=True), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)
    sequence_number = Column(SmallInteger, nullable=False)
    payload = Column(JSONB, nullable=False, default={})
    causal_event_id = Column(UUID(as_uuid=True), nullable=True)

    encounter = relationship("EncounterORM", back_populates="events")


class LabResultORM(Base):
    __tablename__ = "lab_results"

    result_id = Column(UUID(as_uuid=True), primary_key=True)
    encounter_id = Column(UUID(as_uuid=True), ForeignKey("encounters.encounter_id"), nullable=False)
    patient_id = Column(UUID(as_uuid=True), nullable=False)
    collected_at = Column(DateTime(timezone=True), nullable=False)
    resulted_at = Column(DateTime(timezone=True), nullable=False)
    cbc = Column(JSONB, nullable=False)
    bmp = Column(JSONB, nullable=False)
    inflammatory = Column(JSONB, nullable=False)
    disease_specific = Column(JSONB, nullable=False, default={})
    narrative_report = Column(Text, nullable=False)
    critical_flags = Column(ARRAY(Text), nullable=False, default=[])

    encounter = relationship("EncounterORM", back_populates="lab_results")


class ClinicalDocumentORM(Base):
    __tablename__ = "clinical_documents"

    document_id = Column(UUID(as_uuid=True), primary_key=True)
    encounter_id = Column(UUID(as_uuid=True), ForeignKey("encounters.encounter_id"), nullable=False)
    patient_id = Column(UUID(as_uuid=True), nullable=False)
    document_type = Column(String(30), nullable=False)
    authored_at = Column(DateTime(timezone=True), nullable=False)
    author_name = Column(String(100), nullable=False)
    author_role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)

    encounter = relationship("EncounterORM", back_populates="clinical_documents")


class GroundTruthORM(Base):
    __tablename__ = "ground_truth_labels"
    __table_args__ = (UniqueConstraint("encounter_id"),)

    label_id = Column(UUID(as_uuid=True), primary_key=True)
    encounter_id = Column(
        UUID(as_uuid=True), ForeignKey("encounters.encounter_id"), nullable=False, unique=True
    )
    patient_id = Column(UUID(as_uuid=True), nullable=False)
    primary_diagnosis = Column(String(200), nullable=False)
    primary_icd10 = Column(String(10), nullable=False)
    secondary_diagnoses = Column(ARRAY(Text), nullable=False, default=[])
    disease_severity = Column(String(20), nullable=False)
    mortality_risk = Column(Float, nullable=False)
    readmission_risk = Column(Float, nullable=False)
    expected_structured_output = Column(JSONB, nullable=False)
    expected_summary = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    encounter = relationship("EncounterORM", back_populates="ground_truth")


class InferenceResultORM(Base):
    __tablename__ = "inference_results"

    result_id = Column(UUID(as_uuid=True), primary_key=True)
    encounter_id = Column(UUID(as_uuid=True), ForeignKey("encounters.encounter_id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    task = Column(String(50), nullable=False)
    inferred_output = Column(JSONB, nullable=False)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=False)
    ttft_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BenchmarkResultORM(Base):
    __tablename__ = "benchmark_results"

    result_id = Column(UUID(as_uuid=True), primary_key=True)
    run_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    encounter_id = Column(UUID(as_uuid=True), ForeignKey("encounters.encounter_id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    entity_f1 = Column(Float, nullable=True)
    diagnosis_exact = Column(Boolean, nullable=True)
    diagnosis_chapter = Column(Boolean, nullable=True)
    structured_json_score = Column(Float, nullable=True)
    rouge_l = Column(Float, nullable=True)
    risk_brier = Column(Float, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    ttft_ms = Column(Integer, nullable=True)
    tokens_per_sec = Column(Float, nullable=True)
    # LLM quality layer
    json_valid_rate = Column(Float, nullable=True)
    function_calling_success = Column(Float, nullable=True)
    instruction_following_score = Column(Float, nullable=True)
    hallucination_rate = Column(Float, nullable=True)
    evaluated_at = Column(DateTime(timezone=True), server_default=func.now())
