"""JACK-Anbindung: Process-Callback (nur Kopieren), Port-Verwaltung, Rückkanal-Routing.

Signalweg
---------
* JackTrip-Hub (``jacktrip -S -p 5``, kein Auto-Patching) legt je Client einen JACK-Client mit dem
  ``--remotename`` an; seine Ports heißen ``<name>:receive_N`` (Ausgang: Audio vom DJ) und
  ``<name>:send_N`` (Eingang: Audio zum DJ).
* Wir registrieren je DJ zwei Eingangsports ``in_<name>_L/R`` und verbinden ``<name>:receive_*`` damit.
  Der Callback kopiert die Port-Puffer in die Ringe der Engine (Kopie, kein Python-Signalweg dazwischen).
* **Rückkanal ohne Python:** Jeder DJ, der nicht live ist, bekommt per direkter JACK-Verbindung
  ``<live>:receive_N → <dj>:send_N`` den Roh-Feed der aktuell live spielenden (abgebenden) Quelle. Bei
  jeder Übergabe (Statuswechsel ``live``) schaltet der Router um. Der Feed geht an keinem Puffer
  von uns vorbei, nur durch JackTrip selbst.
* Hörer (Rolle ``listener``) bekommen ``program_L/R → <name>:send_N``.

Der Callback läuft ohne Lock und ohne Datenallokation (nur ``np.copyto`` zwischen Port-Array und
Ringpuffer; ``get_array`` liefert eine Sicht auf den Port-Puffer). Er fängt jede Ausnahme und zählt sie.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .config import LiveConfig
from .engine import OUT_PRIME_CHUNKS, Engine, SourceInput

_RX = re.compile(r"^(?P<client>[^:]+):receive_(?P<n>\d+)$")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


@dataclass
class ClientInfo:
    name: str
    role: str
    present: bool = True
    rx: int = 0                      # Zahl der receive_-Ports (Audio vom DJ)
    tx: int = 0                      # Zahl der send_-Ports (Audio zum DJ)
    hears: str | None = None         # "feed:<quelle>" | "program" | None


class JackIO:
    def __init__(self, cfg: LiveConfig, engine: Engine, client_factory: Callable[..., Any] | None = None,
                 mono: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.engine = engine
        self._factory = client_factory
        self._mono = mono
        self.client: Any = None
        self._out: tuple[Any, Any] | None = None
        self._clients: dict[str, ClientInfo] = {}
        self._lock = threading.Lock()           # nur Router/HTTP, nie im Callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.shutdown_reason: str | None = None
        self._prime = OUT_PRIME_CHUNKS * cfg.chunk

    # ------------------------------------------------------------------ Start/Stop

    def start(self) -> None:
        factory = self._factory
        if factory is None:
            import jack
            factory = lambda name: jack.Client(name, no_start_server=True)    # noqa: E731
        c = self.client = factory(self.cfg.client_name)
        if int(c.samplerate) != self.cfg.sample_rate:
            raise RuntimeError(f"JACK läuft mit {c.samplerate} Hz, erwartet {self.cfg.sample_rate} "
                               "(B2B_SAMPLE_RATE / JACK_RATE)")
        self._out = (c.outports.register("program_L"), c.outports.register("program_R"))
        c.set_process_callback(self._process)
        c.set_xrun_callback(self._xrun)
        c.set_shutdown_callback(self._shutdown)
        c.activate()
        self._scan()
        self._thread = threading.Thread(target=self._router, name="b2b-router", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self.client is not None:
            try:
                self.client.deactivate()
                self.client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ Callbacks (Echtzeit)

    def _process(self, frames: int) -> None:
        rt = self.engine.rt
        try:
            base = rt.frames
            rt.stamp = (base, self._mono())
            for si in self.engine.inputs:
                p = si.ports
                if p is None:
                    continue
                if si.origin < 0:
                    si.origin = base
                si.ring.write((p[0].get_array(), p[1].get_array()), frames)
            out = self._out
            ol, orr = out[0].get_array(), out[1].get_array()
            prog = self.engine.program
            eng = self.engine
            if not eng.out_primed and prog.available >= self._prime:
                eng.out_primed = True
            if eng.out_primed:
                got = prog.read((ol, orr), frames)
                if got < frames:
                    ol[got:frames] = 0.0
                    orr[got:frames] = 0.0
                    rt.out_underruns += 1
                    eng.out_primed = False
            else:
                ol[:frames] = 0.0
                orr[:frames] = 0.0
            rt.frames = base + frames
        except Exception:
            rt.cb_errors += 1

    def _xrun(self, delay_us: float) -> None:
        self.engine.rt.xruns += 1

    def _shutdown(self, status: Any, reason: str) -> None:
        self.shutdown_reason = str(reason)

    # ------------------------------------------------------------------ Rollen

    def role_of(self, name: str) -> str:
        return self.cfg.roles.get(name, self.cfg.default_role)

    def set_role(self, name: str, role: str) -> ClientInfo:
        if role not in ("dj", "listener"):
            raise ValueError("Rolle: dj oder listener")
        with self._lock:
            info = self._clients.get(name)
            if info is None:
                raise KeyError(name)
            if info.role == role:
                return info
        if role == "listener":
            self.engine.enable_source(name, False)     # wirft, wenn live
        else:
            self.engine.enable_source(name, True)
        with self._lock:
            info.role = role
            self.cfg.roles[name] = role
        self._connect_all()
        return info

    def clients(self) -> list[dict]:
        live = self.engine.snapshot().get("live")
        with self._lock:
            return [{"name": c.name, "role": c.role, "present": c.present, "live": c.name == live,
                     "rx_ports": c.rx, "tx_ports": c.tx, "hears": c.hears}
                    for c in sorted(self._clients.values(), key=lambda c: c.name)]

    # ------------------------------------------------------------------ Router (eigener Thread)

    def _router(self) -> None:
        while not self._stop.wait(self.cfg.router_interval_s):
            try:
                self._scan()
                self._connect_all()
                self.engine.rt.cpu_load = float(self.client.cpu_load())
            except Exception as e:
                self.engine.last_error = f"router: {type(e).__name__}: {e}"

    def _ports(self, **kw) -> list:
        return list(self.client.get_ports(is_audio=True, **kw))

    def _scan(self) -> None:
        c = self.client
        found: dict[str, int] = {}
        for p in self._ports(name_pattern=r":receive_[0-9]+$", is_output=True):
            m = _RX.match(p.name)
            if m and m["client"] != self.cfg.client_name and m["client"] not in self.cfg.ignore_clients:
                found[m["client"]] = found.get(m["client"], 0) + 1
        sends: dict[str, int] = {}
        for p in self._ports(name_pattern=r":send_[0-9]+$", is_input=True):
            sends[p.name.split(":", 1)[0]] = sends.get(p.name.split(":", 1)[0], 0) + 1
        for name, nrx in found.items():
            with self._lock:
                info = self._clients.get(name)
            if info is None:
                role = self.role_of(name)
                si = SourceInput(name, (c.inports.register(f"in_{_safe(name)}_L"),
                                        c.inports.register(f"in_{_safe(name)}_R")))
                self.engine.attach(si)
                info = ClientInfo(name, role)
                with self._lock:
                    self._clients[name] = info
                if role == "dj":
                    self.engine.enable_source(name, True)
            info.present, info.rx, info.tx = True, nrx, sends.get(name, 0)
        with self._lock:
            for name, info in self._clients.items():
                if name not in found:
                    info.present, info.rx, info.tx = False, 0, 0

    def _connect_all(self) -> None:
        c = self.client
        live = self.engine.snapshot().get("live")
        with self._lock:
            infos = [i for i in self._clients.values() if i.present]
        for info in infos:
            rx = sorted(p.name for p in self._ports(name_pattern=rf"^{re.escape(info.name)}:receive_[0-9]+$",
                                                      is_output=True))
            tx = sorted(p.name for p in self._ports(name_pattern=rf"^{re.escape(info.name)}:send_[0-9]+$",
                                                      is_input=True))
            ins = [f"{self.cfg.client_name}:in_{_safe(info.name)}_L",
                   f"{self.cfg.client_name}:in_{_safe(info.name)}_R"]
            # Eingang: DJ → unsere Ports (nur Rolle dj, Mono-Client auf beide Seiten)
            want_in = {}
            if info.role == "dj" and rx:
                for i, d in enumerate(ins):
                    want_in[d] = [rx[min(i, len(rx) - 1)]]
            for d in ins:
                self._reconcile(d, want_in.get(d, []))
            # Rückkanal: send_N bekommt Feed der live spielenden Quelle bzw. das Programm
            hears = None
            want_tx: dict[str, list[str]] = {t: [] for t in tx}
            if info.role == "listener":
                outs = [f"{self.cfg.client_name}:program_L", f"{self.cfg.client_name}:program_R"]
                for i, t in enumerate(tx):
                    want_tx[t] = [outs[min(i, 1)]]
                hears = "program"
            elif live and live != info.name and live in self._clients and self._clients[live].present:
                lrx = sorted(p.name for p in self._ports(
                    name_pattern=rf"^{re.escape(live)}:receive_[0-9]+$", is_output=True))
                for i, t in enumerate(tx):
                    if lrx:
                        want_tx[t] = [lrx[min(i, len(lrx) - 1)]]
                hears = f"feed:{live}"
            for t, srcs in want_tx.items():
                self._reconcile(t, srcs)
            info.hears = hears if tx else None

    def _reconcile(self, dst: str, want: list[str]) -> None:
        """Verbindungen am Eingangsport ``dst`` auf genau ``want`` bringen (JACK-Ausnahmen werden geschluckt,
        weil ein Client zwischen Scan und Verbinden verschwinden kann)."""
        c = self.client
        try:
            have = {p.name for p in c.get_all_connections(dst)}
        except Exception:
            return
        for src in have - set(want):
            try:
                c.disconnect(src, dst)
            except Exception:
                pass
        for src in set(want) - have:
            try:
                c.connect(src, dst)
            except Exception:
                pass
