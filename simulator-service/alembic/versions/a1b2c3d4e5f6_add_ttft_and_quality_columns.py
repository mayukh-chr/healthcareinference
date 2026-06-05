"""add ttft and quality benchmark columns

Revision ID: a1b2c3d4e5f6
Revises: 5d4f86a28e77
Create Date: 2026-06-03 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op  # type: ignore[attr-defined]

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "5d4f86a28e77"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("inference_results", sa.Column("ttft_ms", sa.Integer(), nullable=True))

    op.add_column("benchmark_results", sa.Column("ttft_ms", sa.Integer(), nullable=True))
    op.add_column("benchmark_results", sa.Column("json_valid_rate", sa.Float(), nullable=True))
    op.add_column("benchmark_results", sa.Column("function_calling_success", sa.Float(), nullable=True))
    op.add_column("benchmark_results", sa.Column("instruction_following_score", sa.Float(), nullable=True))
    op.add_column("benchmark_results", sa.Column("hallucination_rate", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("inference_results", "ttft_ms")
    op.drop_column("benchmark_results", "hallucination_rate")
    op.drop_column("benchmark_results", "instruction_following_score")
    op.drop_column("benchmark_results", "function_calling_success")
    op.drop_column("benchmark_results", "json_valid_rate")
    op.drop_column("benchmark_results", "ttft_ms")
