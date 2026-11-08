"""Test-DJ im Container: spielt Testsignal in JackTrip, hört den Rückkanal, stellt Kicks phasenrichtig ein.

Ablauf je Rolle (Zustand kommt aus ``GET /status`` des Servers und der Datei ``/ctl/next``):

* **live** (Server meldet ``live == NAME``): spielt eigene Kicks + Pad; der Rückkanal ist leer.
* **next** (``/ctl/next`` enthält NAME, nicht live): reicht den gehörten Feed der live spielenden Quelle
  durch (das ist der Master, den der Server um die RTT versetzt zurückfädelt), rastet auf das Raster der
  gehörten Kicks ein und mischt eigene Kicks dazu (Pegel 0→1 in 2 s).
* **sonst**: reicht den Feed durch (damit der Server die RTT schätzen kann), keine eigenen Kicks.
  Nach einer Übergabe blendet der abgebende DJ seine Kicks nach 3 s aus (der Server braucht den Feed der
  abgebenden Quelle bis zum Ende des Übergangs).

Nach der Übernahme blendet der neue Live-DJ den durchgereichten Feed in 2 s aus.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signals as sg      # noqa: E402

FS = sg.FS


class DjCore:
    """Signal und Zustand eines Test-DJs ohne JACK und ohne HTTP (auch für die Simulation in
    ``tests/test_itest_sim.py``). ``process`` ist der Audio-Callback, ``control`` der Steuertakt (0,2 s)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.kick = sg.kick(name)
        # Schwellendurchgang liegt einige Samples nach dem Kick-Anfang: so viel früher einsetzen, damit der
        # eigene Onset auf dem gehörten Onset liegt (das Raster ``grid`` ist aus Onsets bestimmt)
        self.onset_delay = float(sg.detect_onsets(np.concatenate([np.zeros(64, np.float32), self.kick]))[0] - 64)
        self.frames = 0                                  # globaler Frame-Zähler (Beginn des Callbacks)
        self.ring = np.zeros(10 * FS, np.float32)        # Mono-Mitschnitt des gehörten Feeds
        self.state = "idle"                              # idle | next | live | out
        self.prev_state = "idle"
        self.g_feed = 1.0
        self.g_own = 0.0
        self.grid: float | None = None                   # Rasterphase der eigenen Kicks (Frames mod Beat)
        self.last_t = -1e18                              # Zeit (Frames) des zuletzt gespielten Kicks
        self.lock = sg.PhaseLock()
        self.demoted_since: float | None = None
        self.now = 0.0                                   # Uhr des Steuertakts (Sekunden)
        self.pending = np.zeros(len(self.kick) + 4096, np.float32)   # Ausklang laufender Kicks

    # ---- Echtzeit
    def process(self, l_in: np.ndarray, r_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(l_in)
        f0 = self.frames
        x = (l_in + r_in) * 0.5
        w = f0 % len(self.ring)
        first = min(n, len(self.ring) - w)
        self.ring[w:w + first] = x[:first]
        if first < n:
            self.ring[:n - first] = x[first:]
        step = n / (2 * FS)                                  # Rampen: 0 → 1 in 2 s
        g_target = 0.0 if self.state == "live" else 1.0
        self.g_feed += float(np.clip(g_target - self.g_feed, -step, step))
        out_hold = self.state == "out" and (self.now - (self.demoted_since or 0)) < 3.0
        own_target = 1.0 if (self.state in ("live", "next") or out_hold) and self.grid is not None else 0.0
        self.g_own += float(np.clip(own_target - self.g_own, -step, step))
        # Ausklang der Kicks aus den vorigen Blöcken zuerst entnehmen: was dieser Block neu nach
        # ``pending`` schreibt, gehört in die folgenden Blöcke (sonst Vorecho bis zu einem Block vor
        # dem Kick, der Onset rückt im Mittel um einen halben Block nach vorn, S3788).
        tone = self.pending[:n].copy()
        self.pending[:-n] = self.pending[n:]
        self.pending[-n:] = 0.0
        if self.grid is not None:
            g0 = self.grid - self.onset_delay                    # Kick-Anfänge
            lo = max(f0, self.last_t + 0.5 * sg.BEAT)            # nach Zeit statt Index: robust gegen Phasensprung
            k = int(np.ceil((lo - g0) / sg.BEAT))
            while g0 + k * sg.BEAT < f0 + n:
                t = g0 + k * sg.BEAT
                i = int(round(t)) - f0
                if 0 <= i < n:
                    m = min(len(self.kick), n - i)
                    tone[i:i + m] += self.kick[:m]
                    if m < len(self.kick):
                        self.pending[:len(self.kick) - m] += self.kick[m:]
                self.last_t = t
                k += 1
        pd = sg.pad(self.name, f0, n) * (1.0 if self.g_own > 0.0 else 0.0)
        mixed = tone * self.g_own + pd
        self.frames = f0 + n
        return l_in * self.g_feed + mixed, r_in * self.g_feed + mixed

    # ---- Steuerung
    def _grid_from_feed(self) -> None:
        end = self.frames
        n = 8 * FS
        if end < n:
            return
        x = self.ring[np.arange(end - n, end) % len(self.ring)]
        ons = sg.detect_onsets(x) + (end - n)
        if self.lock.update(ons):
            self.grid = self.lock.phase

    def control(self, live: str | None, nxt: str, now: float) -> str | None:
        """Ein Steuertakt: ``live`` laut Server, ``nxt`` aus ``/ctl/next``. Liefert die Zustandsänderung
        als Text (für das Log) oder None."""
        self.now = now
        if live == self.name:
            if self.state != "live" and self.grid is None:
                self.grid = float(self.frames % sg.BEAT)        # erster Live-DJ: eigenes Raster
            self.state = "live"
        elif self.state == "live":
            self.state, self.demoted_since = "out", now
        elif self.state == "out" and now - (self.demoted_since or 0) > 3.0:
            self.state = "next" if nxt == self.name else "idle"
        elif self.state != "out":
            self.state = "next" if nxt == self.name else "idle"
        if self.state in ("next", "idle"):
            self._grid_from_feed()
        if self.state != self.prev_state:
            msg = f"dj {self.name}: {self.prev_state} -> {self.state}"
            self.prev_state = self.state
            return msg
        return None


class Dj:
    def __init__(self, name: str, api: str, ctl: str) -> None:
        import jack
        self.name, self.api, self.ctl = name, api.rstrip("/"), ctl
        self.c = jack.Client(f"dj_{name}", no_start_server=True)
        assert self.c.samplerate == FS, self.c.samplerate
        self.i = [self.c.inports.register(f"in_{k}") for k in (1, 2)]
        self.o = [self.c.outports.register(f"out_{k}") for k in (1, 2)]
        self.core = DjCore(name)
        self.stop = threading.Event()
        self.c.set_process_callback(self._process)

    # ---- Echtzeit
    def _process(self, n: int) -> None:
        lo, ro = self.core.process(self.i[0].get_array(), self.i[1].get_array())
        self.o[0].get_array()[:] = lo
        self.o[1].get_array()[:] = ro

    # ---- Steuerung
    def _status(self) -> dict | None:
        try:
            with urllib.request.urlopen(self.api + "/status", timeout=2) as r:
                return json.loads(r.read())
        except Exception:
            return None

    def _next_name(self) -> str:
        try:
            return open(os.path.join(self.ctl, "next")).read().strip()
        except OSError:
            return ""

    def loop(self) -> None:
        while not self.stop.wait(0.2):
            st = self._status()
            msg = self.core.control(st.get("live") if st else None, self._next_name(), time.monotonic())
            if msg:
                print(msg, flush=True)

    def run(self) -> None:
        self.c.activate()
        self._connect()
        print(f"dj {self.name}: aktiv", flush=True)
        th = threading.Thread(target=self.loop, daemon=True)
        th.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    def _connect(self) -> None:
        deadline = time.time() + 60
        while time.time() < deadline:
            # Namensunabhängig: im Hub-Client-Modus heißt der lokale JackTrip-Client nicht verlässlich
            # "JackTrip" – jeder fremde Client mit receive_/send_-Ports ist JackTrip (S3788).
            # JACK-Muster sind POSIX-ERE – [0-9] statt \\d (vgl. PR #5).
            own = self.c.name + ":"
            rx = [p for p in self.c.get_ports(r":receive_[0-9]+$", is_audio=True, is_output=True)
                  if not p.name.startswith(own)]
            tx = [p for p in self.c.get_ports(r":send_[0-9]+$", is_audio=True, is_input=True)
                  if not p.name.startswith(own)]
            if len(rx) >= 2 and len(tx) >= 2:
                for k in range(2):
                    self.c.connect(rx[k], self.i[k])
                    self.c.connect(self.o[k], tx[k])
                return
            time.sleep(0.5)
        names = [p.name for p in self.c.get_ports()]
        raise SystemExit(f"JackTrip-Ports nicht erschienen; vorhandene Ports: {names}")


if __name__ == "__main__":
    Dj(os.environ["DJ_NAME"], os.environ.get("API_URL", "http://server:8090"),
       os.environ.get("CTL_DIR", "/ctl")).run()
