"""add itl_ms column to inference and benchmark results

Revision ID: c3d4e5f6a7b8
Revises: a1b2c3d4e5f6
Create Date: 2026-06-08 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op  # type: ignore[attr-defined]

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("inference_results", sa.Column("itl_ms", sa.Float(), nullable=True))
    op.add_column("benchmark_results", sa.Column("itl_ms", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("benchmark_results", "itl_ms")
    op.drop_column("inference_results", "itl_ms")
