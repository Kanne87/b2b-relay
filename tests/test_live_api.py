"""HTTP-API (FastAPI-TestClient) gegen den FakeJack-Prüfstand."""

import pytest
from fastapi.testclient import TestClient

from b2b_relay.live.api import create_app
from b2b_relay.live.config import LiveConfig

from live_rig import Rig
from test_live_engine import make_gen, stereo  # noqa: F401  (Fixture)


@pytest.fixture()
def rig_client(stereo):
    rig = Rig(("A", "B", "TV"), cfg=LiveConfig(router_interval_s=0.0, roles={"TV": "listener"}))
    rig.run(22.0, make_gen(stereo))
    return rig, TestClient(create_app(rig.engine, rig.io))


def test_status_contents(rig_client):
    rig, c = rig_client
    r = c.get("/status")
    assert r.status_code == 200
    s = r.json()
    for key in ("live", "buffer", "sources", "drift", "jack", "clients", "uptime_s", "worker"):
        assert key in s
    assert s["live"] == "A"
    assert s["buffer"]["rate"] == pytest.approx(1.0, abs=0.0051) and s["buffer"]["buffer_s"] == pytest.approx(3.0, abs=0.15)
    assert set(s["sources"]["B"]) >= {"rtt_ms", "jitter_ms", "locked", "drift_ppm", "confidence"}
    assert s["jack"]["xruns"] == 0
    roles = {x["name"]: x["role"] for x in s["clients"]}
    assert roles == {"A": "dj", "B": "dj", "TV": "listener"}
    assert c.get("/healthz").status_code == 200


def test_handover_and_errors(rig_client):
    rig, c = rig_client
    assert c.post("/handover", json={"to": "nobody"}).status_code == 404
    assert c.post("/handover", json={"to": "A"}).status_code == 409           # schon live
    assert c.post("/handover", json={"to": "TV"}).status_code == 404          # Hörer ist keine Quelle
    r = c.post("/handover", json={"to": "B"})
    assert r.status_code == 200 and r.json()["accepted"]


def test_roles_endpoint(rig_client):
    rig, c = rig_client
    assert c.post("/roles", json={"client": "B", "role": "listener"}).json() == {"client": "B", "role": "listener"}
    assert c.post("/roles", json={"client": "A", "role": "listener"}).status_code == 409    # live
    assert c.post("/roles", json={"client": "Z", "role": "dj"}).status_code == 404
    assert c.post("/roles", json={"client": "B", "role": "boss"}).status_code == 422


def test_token_required_when_configured(stereo):
    rig = Rig(("A",))
    c = TestClient(create_app(rig.engine, rig.io, token="geheim"))
    assert c.get("/status").status_code == 401
    assert c.get("/status", headers={"Authorization": "Bearer geheim"}).status_code == 200
    assert c.post("/handover", json={"to": "A"}).status_code == 401
