"""Prozess-Einstieg des Live-Servers: ``python -m b2b_relay.live``."""

from __future__ import annotations

import signal
import sys
import threading
import time

from .api import create_app
from .config import LiveConfig
from .engine import Engine
from .jackio import JackIO


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    cfg = LiveConfig.from_env()
    engine = Engine(cfg)
    io = JackIO(cfg, engine)
    io.start()
    engine.start()
    app = create_app(engine, io, cfg.api_token)
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.api_host, port=cfg.api_port,
                                           log_level="warning", access_log=False))
    th = threading.Thread(target=server.run, name="b2b-api", daemon=True)
    th.start()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    print(f"b2b-relay live: JACK-Client {cfg.client_name!r}, API {cfg.api_host}:{cfg.api_port}", flush=True)
    code = 0
    while not stop.is_set():
        if io.shutdown_reason is not None:
            print(f"JACK-Server beendet: {io.shutdown_reason}", file=sys.stderr, flush=True)
            code = 2
            break
        if not th.is_alive():
            print("API-Thread beendet", file=sys.stderr, flush=True)
            code = 3
            break
        time.sleep(0.5)
    server.should_exit = True
    engine.stop()
    io.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
