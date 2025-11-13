### GPU Metrics Backend (FastAPI)

A Python backend service to receive, store, and serve NVIDIA GPU metrics from a C++ monitoring crawler.

#### Features
- HTTP POST /metrics to ingest metrics (single or batch)
- GET /metrics to query historical metrics with filters and optional aggregations (avg/min/max)
- GET /gpus to list available GPU indices
- GET /stats for basic per-GPU statistics
- GET /health health check
- SQLite persistence with simple time-series schema
- File-based ingestion from ws_out.jsonl and msg_out.jsonl (optional)
- Retention policy by age (default 7 days)
- OpenAPI docs at /docs and /redoc

#### Requirements
- Python 3.10+
- pip packages: fastapi, uvicorn, pydantic, sqlalchemy

Install:
```
pip install fastapi uvicorn pydantic sqlalchemy
```

#### Configuration (environment variables)
- PORT: server port (default 8080)
- DATABASE_URL: SQLAlchemy URL (default sqlite:///./metrics.db)
- LOG_LEVEL: logging level (default INFO)
- METRICS_RETENTION_HOURS: retention window in hours (default 168)
- RETENTION_CLEAN_INTERVAL_SEC: retention cleanup frequency (default 3600)
- ENABLE_WS_FILE: true/false to enable reading ws_out.jsonl (default false)
- ENABLE_MSG_FILE: true/false to enable reading msg_out.jsonl (default false)
- INPUT_WS_PATH: path to JSONL file for WS simulation (default ws_out.jsonl)
- INPUT_MSG_PATH: path to JSONL file for messaging simulation (default msg_out.jsonl)
- FILE_POLL_INTERVAL_SEC: polling interval for file readers (default 1.0)

#### Run the server
```
python main.py
```
Then open http://127.0.0.1:8080/ (redirects to /docs). Swagger UI is at /docs and ReDoc is at /redoc.

#### Example payloads
Memory:
```
{
  "timestamp": "2025-01-01T00:00:00Z",
  "gpuIndex": 0,
  "metric": "memory",
  "totalMiB": 10240,
  "usedMiB": 2048,
  "freeMiB": 8192
}
```
Utilization:
```
{"timestamp":"2025-01-01T00:00:00Z","gpuIndex":0,"metric":"utilization","gpuPercent":55,"memoryPercent":12}
```
Temperature:
```
{"timestamp":"2025-01-01T00:00:00Z","gpuIndex":0,"metric":"temperature","celsius":65}
```
Power:
```
{"timestamp":"2025-01-01T00:00:00Z","gpuIndex":0,"metric":"power","watts":115.2}
```
Clocks:
```
{"timestamp":"2025-01-01T00:00:00Z","gpuIndex":0,"metric":"clocks","graphicsMHz":1800,"memoryMHz":7000}
```
Process:
```
{"timestamp":"2025-01-01T00:00:00Z","gpuIndex":0,"metric":"process","pid":1234,"usedMiB":512}
```

Batching: send an array of objects to /metrics.

#### Query examples
- GET /metrics?gpuIndex=0&metric=memory&start=2025-01-01T00:00:00Z&end=2025-01-01T01:00:00Z
- GET /metrics?metric=utilization&aggregate=avg&field=gpuPercent
- GET /metrics?metric=power&aggregate=max&field=watts
- GET /gpus
- GET /stats
- GET /health

#### Tests
Install dev deps and run pytest:
```
pip install pytest httpx
pytest -q
```

#### Notes
- Timestamps are stored as UTC; if incoming timestamp lacks timezone, UTC is assumed.
- Retention deletes rows older than METRICS_RETENTION_HOURS based on the metric timestamp.
- File-based ingestion expects one JSON metric per line in the configured files.
