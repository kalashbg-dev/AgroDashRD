from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agrodash.infra.db import Base, enum_column


class SourceLayout(StrEnum):
    """Shape of a price report: wholesale is one price per row, retail one column per market."""

    WHOLESALE = "wholesale"
    RETAIL = "retail"


class IngestionTrigger(StrEnum):
    DRIVE = "drive"
    UPLOAD = "upload"
    FILE = "file"


class IngestionStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RejectionReason(StrEnum):
    INVALID_DATE = "invalid_date"
    INVALID_PRICE = "invalid_price"
    UNKNOWN_UNIT = "unknown_unit"
    UNIT_MISMATCH = "unit_mismatch"
    UNEXPECTED_LAYOUT = "unexpected_layout"
    DUPLICATE_PRICE = "duplicate_price"


class IngestionRun(Base):
    """One attempt to load a source file. A file (by content hash) succeeds at most once."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (
        Index(
            "uq_ingestion_runs_succeeded_file",
            "file_sha256",
            unique=True,
            sqlite_where=text("status = 'succeeded'"),
            postgresql_where=text("status = 'succeeded'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255))
    file_sha256: Mapped[str] = mapped_column(String(64), index=True)
    layout: Mapped[SourceLayout] = mapped_column(enum_column(SourceLayout, "source_layout"))
    trigger: Mapped[IngestionTrigger] = mapped_column(
        enum_column(IngestionTrigger, "ingestion_trigger")
    )
    status: Mapped[IngestionStatus] = mapped_column(
        enum_column(IngestionStatus, "ingestion_status"), default=IngestionStatus.RUNNING
    )
    started_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    rows_read: Mapped[int] = mapped_column(default=0)
    prices_loaded: Mapped[int] = mapped_column(default=0)
    rows_rejected: Mapped[int] = mapped_column(default=0)
    # Cells marked "N/D" or left blank: no observation, not an error.
    values_missing: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(String(500))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    quarantined_rows: Mapped[list["QuarantinedRow"]] = relationship(back_populates="run")


class QuarantinedRow(Base):
    """A source row that could not be loaded, kept with its reason for review."""

    __tablename__ = "quarantined_rows"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("ingestion_runs.id"), index=True)
    row_number: Mapped[int]
    reason: Mapped[RejectionReason] = mapped_column(
        enum_column(RejectionReason, "rejection_reason")
    )
    detail: Mapped[str | None] = mapped_column(String(200))
    # The row's cells as a JSON array, exactly as read.
    raw_values: Mapped[str] = mapped_column(Text)

    run: Mapped[IngestionRun] = relationship(back_populates="quarantined_rows")
