from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import func, select

from core import (
    MetricORM,
    MetricOut,
    MetricType,
    db_session,
    persist_metric,
    MemoryMetric,
    UtilizationMetric,
    TemperatureMetric,
    PowerMetric,
    ClocksMetric,
    ProcessMetric,
    IncomingMetric,
    logger,
)

router = APIRouter()


class Aggregation(str, Enum):
    avg = "avg"
    min = "min"
    max = "max"


@router.get("/metrics")
async def get_metrics(
    metric: Optional[MetricType] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    limit: int = Query(500, ge=1, le=10000),
    order: Literal["asc", "desc"] = Query("desc"),
    aggregate: Optional[Aggregation] = Query(None),
    field: Optional[str] = Query(None, description="Field name to aggregate (e.g., usedMiB, gpuPercent, watts)"),
    hostname: Optional[str] = Query(None, description="Filter by hostname"),
    gpuId: Optional[str] = Query(None, description="Filter by GPU ID"),
    gpuName: Optional[str] = Query(None, description="Filter by GPU Name"),
):
    # Normalize times
    if start and start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    with db_session() as s:
        q = select(MetricORM)
        if metric is not None:
            q = q.where(MetricORM.metric_type == metric.value)
        if hostname is not None:
            q = q.where(MetricORM.hostname == hostname)
        if gpuId is not None:
            q = q.where(MetricORM.gpu_id == gpuId)
        if gpuName is not None:
            q = q.where(MetricORM.gpu_name == gpuName)
        if start is not None:
            q = q.where(MetricORM.timestamp >= start)
        if end is not None:
            q = q.where(MetricORM.timestamp <= end)

        if aggregate:
            # Aggregate over the unified numeric column
            col = MetricORM.value

            if aggregate == Aggregation.avg:
                sel = select(func.avg(col))
            elif aggregate == Aggregation.min:
                sel = select(func.min(col))
            else:
                sel = select(func.max(col))

            # apply filters (re-apply where clauses)
            for crit in q._where_criteria:  # type: ignore[attr-defined]
                sel = sel.where(crit)
            val = s.execute(sel).scalar_one_or_none()
            return {"aggregate": aggregate.value, "field": field, "value": val}

        # non-aggregate path
        q = q.order_by(MetricORM.timestamp.asc() if order == "asc" else MetricORM.timestamp.desc()).limit(limit)
        rows = s.execute(q).scalars().all()
        items = [
            MetricOut(
                id=r.id,
                timestamp=r.timestamp,
                received_at=r.received_at,
                metric=MetricType(r.metric_type),
                payload=r.payload,
            ).model_dump()
            for r in rows
        ]

    return {
        "items": items,
        "count": len(items),
    }


# Shared ingestion logic so both HTTP POST and WebSocket can use it
async def _ingest_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed: List[IncomingMetric] = []
    errors: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        try:
            # Discriminate on 'metric' (case-insensitive) and pre-process payloads
            raw_metric = item.get("metric")
            metric_key = str(raw_metric).strip().lower() if isinstance(raw_metric, str) else raw_metric

            # Fill derived/missing fields for memory metric
            if metric_key == MetricType.memory.value:
                if "freeMiB" not in item and "totalMiB" in item and "usedMiB" in item:
                    try:
                        free_val = int(float(item["totalMiB"]) - float(item["usedMiB"]))
                        if free_val < 0:
                            free_val = 0
                        item = {**item, "freeMiB": free_val}
                    except Exception:
                        # ignore, let validation handle
                        pass

            # Normalize client payload differences for other metrics
            if metric_key == MetricType.temperature.value:
                # Support gpuCelsius -> celsius
                if "celsius" not in item and "gpuCelsius" in item:
                    try:
                        item = {**item, "celsius": int(item["gpuCelsius"])}
                    except Exception:
                        pass
            elif metric_key == MetricType.power.value:
                # Support milliwatts -> watts
                if "watts" not in item and "milliwatts" in item:
                    try:
                        mw = float(item["milliwatts"])
                        item = {**item, "watts": mw / 1000.0}
                    except Exception:
                        pass
            elif metric_key == MetricType.clocks.value:
                # Support memMHz -> memoryMHz
                if "memoryMHz" not in item and "memMHz" in item:
                    try:
                        item = {**item, "memoryMHz": int(item["memMHz"])}
                    except Exception:
                        pass
                # smMHz is supported by the model as optional; no change needed

            if metric_key == MetricType.memory.value:
                parsed.append(MemoryMetric(**item))
            elif metric_key == MetricType.utilization.value:
                parsed.append(UtilizationMetric(**item))
            elif metric_key == MetricType.temperature.value:
                parsed.append(TemperatureMetric(**item))
            elif metric_key == MetricType.power.value:
                parsed.append(PowerMetric(**item))
            elif metric_key == MetricType.clocks.value:
                parsed.append(ClocksMetric(**item))
            elif metric_key == MetricType.process.value:
                parsed.append(ProcessMetric(**item))
            else:
                raise ValidationError.from_exception_data("IncomingMetric", [
                    {
                        "type": "value_error",
                        "loc": (idx, "metric"),
                        "msg": f"Unsupported metric type: {raw_metric}",
                        "input": raw_metric,
                    }
                ])
        except ValidationError as e:
            errors.append({"index": idx, "errors": json.loads(e.json())})

    if errors and not parsed:
        # mirror HTTP error structure for WS caller to interpret
        raise HTTPException(status_code=422, detail={"errors": errors})

    saved_ids: List[int] = []
    with db_session() as s:
        for m in parsed:
            row = persist_metric(s, m)
            saved_ids.append(row.id)

    response: Dict[str, Any] = {"saved": len(saved_ids), "ids": saved_ids}
    if errors:
        response["errors"] = errors
    return response


@router.post("/metrics", status_code=201)
async def post_metrics(request: Request):
    body = await request.json()

    # Accept single object or list of objects
    items: List[Dict[str, Any]]
    if isinstance(body, list):
        items = body
    elif isinstance(body, dict):
        items = [body]
    else:
        raise HTTPException(status_code=400, detail="Body must be a JSON object or array of objects")

    response = await _ingest_items(items)
    return JSONResponse(status_code=201, content=response)


@router.websocket("/ws/metrics")
async def metrics_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Receive text frames as JSON
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await ws.send_json({"error": "Invalid JSON"})
                continue

            # Normalize to list of dicts
            items: List[Dict[str, Any]]
            if isinstance(data, list):
                if not all(isinstance(x, dict) for x in data):
                    await ws.send_json({"error": "Array items must be objects"})
                    continue
                items = data  # type: ignore[assignment]
            elif isinstance(data, dict):
                items = [data]
            else:
                await ws.send_json({"error": "Payload must be an object or array of objects"})
                continue

            # Process and persist
            try:
                resp = await _ingest_items(items)
                # For WS, include type for clarity
                await ws.send_json({"type": "ack", **resp})
            except HTTPException as he:
                # send structured error
                await ws.send_json({"type": "error", "status": he.status_code, "detail": he.detail})
    except WebSocketDisconnect:
        # Client disconnected; just exit handler
        return
    except Exception as e:
        # Keep the connection stable on unexpected server errors
        try:
            await ws.send_json({"type": "error", "status": 500, "detail": str(e)})
        except Exception:
            pass
        # continue loop on next iteration by recursion-like behavior is not ideal; simply return to close
        # However, to avoid reconnect loops initiated by server, we won't close explicitly here.
        return
