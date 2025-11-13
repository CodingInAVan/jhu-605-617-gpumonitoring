from fastapi.testclient import TestClient
from main import app, init_db

client = TestClient(app)


def setup_module(module):
    # ensure DB tables exist
    init_db()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_ws_and_query_memory_metric():
    payload = {
        "timestamp": "2025-01-01T00:00:00Z",
        "metric": "memory",
        "totalMiB": 12288,
        "usedMiB": 4096,
        "freeMiB": 8192,
    }
    with client.websocket_connect("/ws/metrics") as ws:
        ws.send_json(payload)
        ack = ws.receive_json()
        assert ack.get("type") == "ack"
        assert ack.get("saved", 0) >= 1

    r2 = client.get("/metrics", params={"metric": "memory", "limit": 10})
    assert r2.status_code == 200
    items = r2.json()["items"]
    assert any(itm["payload"]["usedMiB"] == 4096 for itm in items)


def test_aggregate():
    # add a couple utilization points
    p1 = {
        "timestamp": "2025-01-01T00:00:00Z",
        "metric": "utilization",
        "gpuPercent": 50,
        "memoryPercent": 10,
    }
    p2 = {
        "timestamp": "2025-01-01T00:10:00Z",
        "metric": "utilization",
        "gpuPercent": 70,
        "memoryPercent": 20,
    }
    with client.websocket_connect("/ws/metrics") as ws:
        ws.send_json(p1)
        ack1 = ws.receive_json()
        assert ack1.get("type") == "ack"
        ws.send_json(p2)
        ack2 = ws.receive_json()
        assert ack2.get("type") == "ack"

    r = client.get("/metrics", params={"metric": "utilization", "gpuIndex": 2, "aggregate": "avg", "field": "gpuPercent"})
    assert r.status_code == 200
    val = r.json()["value"]
    assert 59.9 < val < 60.1
