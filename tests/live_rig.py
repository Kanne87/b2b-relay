"""Prüfstand für das Live-Modul ohne jackd: FakeJack + Engine im selben Thread (deterministisch)."""

from __future__ import annotations

import numpy as np

from b2b_relay.live.config import LiveConfig
from b2b_relay.live.engine import Engine
from b2b_relay.live.jackio import JackIO
from fakejack import FakeJack

FS = 48000
PERIOD = 256


class Rig:
    def __init__(self, clients=("A", "B"), cfg: LiveConfig | None = None, clock_ppm: float = 0.0) -> None:
        self.cfg = cfg or LiveConfig(router_interval_s=0.0)
        self.world = FakeJack(PERIOD, FS)
        for c in clients:
            self.world.add_remote(c)
        self.now = 1000.0                      # CLOCK_MONOTONIC (simuliert)
        self.ppm = clock_ppm                   # JACK-Frames laufen um ppm schneller als die Wanduhr
        self.engine = Engine(self.cfg, mono=lambda: self.now)
        self.io = JackIO(self.cfg, self.engine, self.world.factory, mono=lambda: self.now)
        self.io.start()                        # Thread des Routers läuft nicht ins Gewicht: wir rufen selbst
        self.io._stop.set()
        self.engine.proc = 0
        self.engine._started_mono = self.now
        self.cyc = 0
        self.sent: dict[str, list[np.ndarray]] = {}
        self.out: list[np.ndarray] = []

    def feed(self, name: str, l: np.ndarray, r: np.ndarray) -> None:
        self.world.ports[f"{name}:receive_1"].buf[:] = l
        self.world.ports[f"{name}:receive_2"].buf[:] = r

    def _propagate(self) -> None:
        w = self.world
        acc: dict[str, np.ndarray] = {}
        for a, b in w.conns:
            if a in w.ports and b in w.ports:
                acc[b] = acc.get(b, 0) + w.ports[a].buf
        for b, x in acc.items():
            w.ports[b].buf[:] = x

    def cycle(self) -> None:
        w = self.world
        self._propagate()                      # Remote-receive → unsere Eingänge
        self.io._process(PERIOD)
        # Programmausgang und Rückkanäle → send-Ports; Mitschnitt der Ausgabe
        self._propagate()
        self.out.append(np.stack([w.ports["b2b-relay:program_L"].buf,
                                  w.ports["b2b-relay:program_R"].buf]).copy())
        for n in {p.split(":")[0] for p in w.ports if p.endswith(":send_1")}:
            self.sent.setdefault(n, []).append(w.ports[f"{n}:send_1"].buf.copy())
        self.cyc += 1
        self.now += PERIOD / FS * (1.0 - self.ppm * 1e-6)
        if not (self.engine._thread and self.engine._thread.is_alive()):
            while self.engine.pump():
                pass
        if self.cyc % 20 == 0:
            self.io._scan()
            self.io._connect_all()

    def run(self, seconds: float, gen) -> None:
        """``gen(start_frame, n)`` → {name: (L, R)} liefert die Eingangsblöcke."""
        for _ in range(int(seconds * FS / PERIOD)):
            for name, (l, r) in gen(self.cyc * PERIOD, PERIOD).items():
                if f"{name}:receive_1" in self.world.ports:       # fehlende Clients (Tests) auslassen
                    self.feed(name, l, r)
            self.cycle()

    def program(self) -> np.ndarray:
        return np.concatenate(self.out, axis=1)
