from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from core import db_session, MetricORM, logger

router = APIRouter()


@router.get("/health")
async def health() -> Dict[str, Any]:
    try:
        with db_session() as s:
            total = s.execute(select(func.count(MetricORM.id))).scalar_one()
        return {"status": "ok", "database": "ok", "totalMetrics": total}
    except SQLAlchemyError as e:
        logger.exception("Health check DB error")
        raise HTTPException(status_code=500, detail=str(e))
