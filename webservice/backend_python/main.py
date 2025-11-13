from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import (
    JSON as SA_JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    create_engine,
    delete,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Import shared core (single source of truth for DB/models)
from core import (
    init_db as core_init_db,
    db_session as core_db_session,
    persist_metric as core_persist_metric,
    MetricType,
    MemoryMetric,
    UtilizationMetric,
    TemperatureMetric,
    PowerMetric,
    ClocksMetric,
    ProcessMetric,
)

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
    gpu_index = Column(Integer, index=True, nullable=False)
    metric_type = Column(String(32), index=True, nullable=False)
    # Keep entire original payload for flexibility
    payload = Column(SA_JSON, nullable=False)
    # A few generic numeric columns to support simple aggregations fast
    num1 = Column(Integer, nullable=True)  # e.g., usedMiB, gpuPercent, celsius, graphicsMHz
    num2 = Column(Integer, nullable=True)  # e.g., memoryPercent, memoryMHz, freeMiB, totalMiB
    numf = Column(String(64), nullable=True)  # store float as text (for cross DB), e.g., watts
    pid = Column(Integer, nullable=True)


# -------------------------------------------------
# Pydantic models for input validation (discriminated union on 'metric')
# -------------------------------------------------
class BaseMetric(BaseModel):
    timestamp: datetime
    gpuIndex: int = Field(..., ge=0)
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
    graphicsMHz: int = Field(..., ge=0)
    memoryMHz: int = Field(..., ge=0)


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
    gpuIndex: int
    metric: MetricType
    payload: Dict[str, Any]


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="GPU Metrics Backend",
    description="Receives, stores, and serves GPU metrics from NVIDIA monitoring crawlers.",
    version="1.0.0",
)


def init_db() -> None:
    # Delegate to shared core module
    core_init_db()


@contextmanager
def db_session() -> Iterable[Session]:
    # Delegate to shared core session
    with core_db_session() as session:
        yield session


# -------------------------------------------------
# Persistence helpers
# -------------------------------------------------
NUMERIC_FIELD_HINTS = {
    MetricType.memory: ("usedMiB", "totalMiB", None, None),
    MetricType.utilization: ("gpuPercent", "memoryPercent", None, None),
    MetricType.temperature: ("celsius", None, None, None),
    MetricType.power: ("watts", None, "float", None),
    MetricType.clocks: ("graphicsMHz", "memoryMHz", None, None),
    MetricType.process: ("usedMiB", None, None, "pid"),
}


def persist_metric(session: Session, m: IncomingMetric, received_at: Optional[datetime] = None) -> MetricORM:
    # Delegate to shared core implementation
    return core_persist_metric(session, m, received_at)


# -------------------------------------------------
# Routes (mounted via controllers)
# -------------------------------------------------
from controllers.health import router as health_router
from controllers.gpus import router as gpus_router
from controllers.metrics import router as metrics_router
from controllers.stats import router as stats_router

app.include_router(health_router)
app.include_router(gpus_router)
app.include_router(metrics_router)
app.include_router(stats_router)


# -------------------------------------------------
# File-based input readers (simulate WS and messaging)
# -------------------------------------------------
class FileTailer(threading.Thread):
    def __init__(self, file_path: str, name: str):
        super().__init__(daemon=True, name=name)
        self.file_path = Path(file_path)
        self._stop = threading.Event()
        self._pos = 0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        logger.info(f"Starting tailer for {self.file_path}")
        while not self._stop.is_set():
            try:
                if not self.file_path.exists():
                    time.sleep(FILE_POLL_INTERVAL_SEC)
                    continue
                with self.file_path.open("r", encoding="utf-8") as f:
                    f.seek(self._pos)
                    for line in f:
                        self._pos = f.tell()
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            # Reuse POST logic by validating and persisting directly
                            metric_type = obj.get("metric")
                            m: IncomingMetric
                            if metric_type == MetricType.memory.value:
                                m = MemoryMetric(**obj)
                            elif metric_type == MetricType.utilization.value:
                                m = UtilizationMetric(**obj)
                            elif metric_type == MetricType.temperature.value:
                                m = TemperatureMetric(**obj)
                            elif metric_type == MetricType.power.value:
                                m = PowerMetric(**obj)
                            elif metric_type == MetricType.clocks.value:
                                m = ClocksMetric(**obj)
                            elif metric_type == MetricType.process.value:
                                m = ProcessMetric(**obj)
                            else:
                                logger.warning(f"Skipping unsupported metric type in file: {metric_type}")
                                continue
                            with db_session() as s:
                                persist_metric(s, m)
                        except Exception as e:
                            logger.exception(f"Error processing line from {self.file_path}: {e}")
                time.sleep(FILE_POLL_INTERVAL_SEC)
            except Exception:
                logger.exception(f"Tailer failure for {self.file_path}")
                time.sleep(FILE_POLL_INTERVAL_SEC)


# -------------------------------------------------
# Retention policy task
# -------------------------------------------------
class RetentionWorker(threading.Thread):
    def __init__(self, hours: int):
        super().__init__(daemon=True, name="retention-worker")
        self.hours = hours
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        if self.hours <= 0:
            logger.info("Retention disabled (hours <= 0)")
            return
        delta = timedelta(hours=self.hours)
        while not self._stop.is_set():
            try:
                cutoff = datetime.now(timezone.utc) - delta
                with db_session() as s:
                    res = s.execute(delete(MetricORM).where(MetricORM.timestamp < cutoff))
                    deleted = res.rowcount or 0
                if deleted:
                    logger.info(f"Retention removed {deleted} rows older than {cutoff.isoformat()}")
            except Exception:
                logger.exception("Retention worker error")
            # Sleep until next cycle
            for _ in range(int(RETENTION_CLEAN_INTERVAL_SEC)):
                if self._stop.is_set():
                    break
                time.sleep(1)


# -------------------------------------------------
# Lifecycle events (FastAPI lifespan)
# -------------------------------------------------
_tailer_threads: List[FileTailer] = []
_retention_thread: Optional[RetentionWorker] = None
_started_at = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan_ctx(app: FastAPI):
    # Startup
    init_db()
    # Start file readers if enabled
    if ENABLE_WS_FILE:
        _tailer_threads.append(FileTailer(INPUT_WS_PATH, name="ws-file-tailer"))
    if ENABLE_MSG_FILE:
        _tailer_threads.append(FileTailer(INPUT_MSG_PATH, name="msg-file-tailer"))
    for t in _tailer_threads:
        t.start()
    # Start retention worker
    global _retention_thread
    _retention_thread = RetentionWorker(RETENTION_HOURS)
    _retention_thread.start()
    logger.info("Backend started")

    try:
        yield
    finally:
        # Shutdown
        for t in _tailer_threads:
            t.stop()
        if _retention_thread:
            _retention_thread.stop()


# Register lifespan handler to avoid deprecated on_event usage
app.router.lifespan_context = lifespan_ctx


# Root endpoint and simple greeting kept for compatibility
@app.get("/")
async def root():
    # Redirect to Swagger UI for quick API exploration
    return RedirectResponse(url="/docs")


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


# Convenience: run with `python main.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
