from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fastapi import APIRouter
from sqlalchemy import func, select

from core import MetricORM, db_session

router = APIRouter()


@router.get("/gpus")
async def list_gpus() -> Dict[str, Any]:
    # List distinct GPU identifiers (hostname, gpuId, gpuName) with latest sample timestamp
    with db_session() as s:
        rows: List[Tuple[str, str, str, object]] = s.execute(
            select(
                MetricORM.hostname,
                MetricORM.gpu_id,
                MetricORM.gpu_name,
                func.max(MetricORM.timestamp),
            ).group_by(MetricORM.hostname, MetricORM.gpu_id, MetricORM.gpu_name)
        ).all()

    gpus = [
        {
            "hostname": h,
            "gpuId": gid,
            "gpuName": gname,
            "latest": (ts.isoformat() if ts else None),
        }
        for h, gid, gname, ts in rows
    ]
    return {"gpus": gpus}
