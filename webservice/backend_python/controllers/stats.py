from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from sqlalchemy import func, select

from core import MetricORM, db_session

router = APIRouter()


@router.get("/stats")
async def stats():
    # Return totals per metric_type (gpuIndex removed)
    with db_session() as s:
        q = select(
            MetricORM.metric_type,
            func.count(MetricORM.id),
            func.max(MetricORM.timestamp),
        ).group_by(MetricORM.metric_type)
        rows = s.execute(q).all()
    out: Dict[str, Any] = {mtype: {"count": cnt, "latest": (latest.isoformat() if latest else None)} for mtype, cnt, latest in rows}
    return {"metrics": out}
