from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from sqlalchemy import (
    JSON as SA_JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Index,
    Float,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# -------------------------------------------------
# Configuration via environment variables
# -------------------------------------------------
PORT: int = int(os.getenv("PORT", "8080"))
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./metrics.db")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
RETENTION_HOURS: int = int(os.getenv("METRICS_RETENTION_HOURS", "168"))  # default 7 days
ENABLE_WS_FILE: bool = os.getenv("ENABLE_WS_FILE", "false").lower() in {"1", "true", "yes"}
ENABLE_MSG_FILE: bool = os.getenv("ENABLE_MSG_FILE", "false").lower() in {"1", "true", "yes"}
INPUT_WS_PATH: str = os.getenv("INPUT_WS_PATH", "ws_out.jsonl")
INPUT_MSG_PATH: str = os.getenv("INPUT_MSG_PATH", "msg_out.jsonl")
RETENTION_CLEAN_INTERVAL_SEC: int = int(os.getenv("RETENTION_CLEAN_INTERVAL_SEC", "3600"))
FILE_POLL_INTERVAL_SEC: float = float(os.getenv("FILE_POLL_INTERVAL_SEC", "1.0"))

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("gpu-metrics-backend")

# -------------------------------------------------
# Database setup (SQLAlchemy)
# -------------------------------------------------
engine: Engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
metadata = MetaData()
Base = declarative_base(metadata=metadata)


class MetricType(str, Enum):
    memory = "memory"
    utilization = "utilization"
    temperature = "temperature"
    power = "power"
    clocks = "clocks"
    process = "process"


class MetricORM(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    received_at = Column(DateTime(timezone=True), index=True, nullable=False)
    metric_type = Column(String(32), index=True, nullable=False)
    # New combined key components
    hostname = Column(String(255), index=True, nullable=True)
    gpu_id = Column(String(255), index=True, nullable=True)
    gpu_name = Column(String(255), index=True, nullable=True)
    __table_args__ = (
        Index("ix_metrics_host_gpu", "hostname", "gpu_id", "gpu_name"),
    )
    # Keep entire original payload for flexibility
    payload = Column(SA_JSON, nullable=False)
    # Generic numeric value + unit for aggregation
    value = Column(Float, nullable=True)  # primary numeric value (e.g., usedMiB, gpuPercent, celsius, watts, graphicsMHz)
    unit = Column(String(32), nullable=True)  # e.g., MiB, %, C, W, MHz
    pid = Column(Integer, nullable=True)


# -------------------------------------------------
# Pydantic models for input validation (discriminated union on 'metric')
# -------------------------------------------------
class BaseMetric(BaseModel):
    timestamp: datetime
    hostname: Optional[str] = None
    gpuName: Optional[str] = None
    gpuId: Optional[str] = None
    metric: MetricType

    @field_validator("timestamp")
    @classmethod
    def ensure_tzaware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            # assume UTC if tz missing
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class MemoryMetric(BaseMetric):
    metric: Literal[MetricType.memory]
    totalMiB: int = Field(..., ge=0)
    usedMiB: int = Field(..., ge=0)
    freeMiB: int = Field(..., ge=0)


class UtilizationMetric(BaseMetric):
    metric: Literal[MetricType.utilization]
    gpuPercent: float = Field(..., ge=0, le=100)
    memoryPercent: float = Field(..., ge=0, le=100)


class TemperatureMetric(BaseMetric):
    metric: Literal[MetricType.temperature]
    celsius: int


class PowerMetric(BaseMetric):
    metric: Literal[MetricType.power]
    watts: float = Field(..., ge=0)


class ClocksMetric(BaseMetric):
    metric: Literal[MetricType.clocks]
    graphicsMHz: Optional[int] = Field(None, ge=0)
    memoryMHz: Optional[int] = Field(None, ge=0)
    smMHz: Optional[int] = Field(None, ge=0)

    @model_validator(mode="after")
    def at_least_one_clock(cls, values):
        # Pydantic v2 model_validator with mode="after" receives the model instance
        # but here we accept a dict-like in case of compatibility; access via getattr
        g = getattr(values, 'graphicsMHz', None)
        m = getattr(values, 'memoryMHz', None)
        s = getattr(values, 'smMHz', None)
        if g is None and m is None and s is None:
            raise ValueError("At least one of graphicsMHz, memoryMHz, or smMHz must be provided")
        return values


class ProcessMetric(BaseMetric):
    metric: Literal[MetricType.process]
    pid: Optional[int] = Field(None)
    usedMiB: int = Field(..., ge=0)


IncomingMetric = Union[
    MemoryMetric,
    UtilizationMetric,
    TemperatureMetric,
    PowerMetric,
    ClocksMetric,
    ProcessMetric,
]


# Response model
class MetricOut(BaseModel):
    id: int
    timestamp: datetime
    received_at: datetime
    metric: MetricType
    payload: Dict[str, Any]


@contextmanager
def db_session() -> Iterable[Session]:
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    # For SQLite/testing, rebuild schema to reflect model changes
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


# -------------------------------------------------
# Persistence helpers
# -------------------------------------------------

def persist_metric(session: Session, m: IncomingMetric, received_at: Optional[datetime] = None) -> MetricORM:
    received = received_at or datetime.now(timezone.utc)
    payload = json.loads(m.model_dump_json())  # retain camelCase keys

    # Determine primary numeric value and unit based on metric type
    value: Optional[float] = None
    unit: Optional[str] = None
    pid: Optional[int] = None

    if m.metric == MetricType.memory:
        # store usedMiB as primary value
        v = payload.get("usedMiB")
        if isinstance(v, (int, float)):
            value = float(v)
            unit = "MiB"
    elif m.metric == MetricType.utilization:
        v = payload.get("gpuPercent")
        if isinstance(v, (int, float)):
            value = float(v)
            unit = "%"
    elif m.metric == MetricType.temperature:
        v = payload.get("celsius")
        if isinstance(v, (int, float)):
            value = float(v)
            unit = "C"
    elif m.metric == MetricType.power:
        v = payload.get("watts")
        if isinstance(v, (int, float)):
            value = float(v)
            unit = "W"
    elif m.metric == MetricType.clocks:
        # prefer graphicsMHz if present, else memoryMHz, else smMHz if client sends
        for key in ("graphicsMHz", "memoryMHz", "smMHz"):
            v = payload.get(key)
            if isinstance(v, (int, float)):
                value = float(v)
                unit = "MHz"
                break
    elif m.metric == MetricType.process:
        v = payload.get("usedMiB")
        if isinstance(v, (int, float)):
            value = float(v)
            unit = "MiB"
        if payload.get("pid") is not None:
            try:
                pid = int(payload.get("pid"))
            except Exception:
                pid = None

    row = MetricORM(
        timestamp=m.timestamp,
        received_at=received,
        metric_type=m.metric.value,
        hostname=payload.get("hostname"),
        gpu_id=payload.get("gpuId"),
        gpu_name=payload.get("gpuName"),
        payload=payload,
        value=value,
        unit=unit,
        pid=pid,
    )
    session.add(row)
    session.flush()
    return row


# Re-export commonly used names for convenience
__all__ = [
    "PORT",
    "DATABASE_URL",
    "RETENTION_HOURS",
    "ENABLE_WS_FILE",
    "ENABLE_MSG_FILE",
    "INPUT_WS_PATH",
    "INPUT_MSG_PATH",
    "RETENTION_CLEAN_INTERVAL_SEC",
    "FILE_POLL_INTERVAL_SEC",
    "logger",
    "engine",
    "SessionLocal",
    "Base",
    "MetricType",
    "MetricORM",
    "BaseMetric",
    "MemoryMetric",
    "UtilizationMetric",
    "TemperatureMetric",
    "PowerMetric",
    "ClocksMetric",
    "ProcessMetric",
    "IncomingMetric",
    "MetricOut",
    "db_session",
    "init_db",
    "persist_metric",
    "ValidationError",
]
