"""HTTP-Steuerung (FastAPI) im selben Prozess wie die Engine. Keine Weboberfläche (Scheibe 3)."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .engine import Engine
from .jackio import JackIO


class HandoverBody(BaseModel):
    to: str


class RoleBody(BaseModel):
    client: str
    role: str


def create_app(engine: Engine, io: JackIO, token: str = "") -> FastAPI:
    app = FastAPI(title="b2b-relay", version="0.2.0")

    def auth(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "Bearer-Token fehlt oder falsch")

    @app.get("/healthz")
    def healthz() -> dict:
        snap = engine.snapshot()
        ok = bool(snap.get("started")) and io.shutdown_reason is None
        if not ok:
            raise HTTPException(503, io.shutdown_reason or "Engine noch nicht gestartet")
        return {"ok": True}

    @app.get("/status", dependencies=[Depends(auth)])
    def status() -> dict:
        snap = dict(engine.snapshot())
        snap["clients"] = io.clients()
        snap["jack"] = {**snap.get("jack", {}), "shutdown": io.shutdown_reason}
        return snap

    @app.post("/handover", dependencies=[Depends(auth)])
    def handover(body: HandoverBody) -> dict[str, Any]:
        res = engine.handover(body.to)
        if not res["accepted"]:
            raise HTTPException(409 if res["reason"] != "unknown_source" else 404, res)
        return res

    @app.post("/roles", dependencies=[Depends(auth)])
    def roles(body: RoleBody) -> dict[str, Any]:
        if body.role not in ("dj", "listener"):
            raise HTTPException(422, "role: dj oder listener")
        try:
            info = io.set_role(body.client, body.role)
        except KeyError:
            raise HTTPException(404, f"Client {body.client!r} ist nicht verbunden")
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"client": info.name, "role": info.role}

    return app
