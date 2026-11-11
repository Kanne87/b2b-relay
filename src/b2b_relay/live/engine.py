"""Live-Engine: Worker-Thread zwischen JACK-Callback und RelayMixer.

Der JACK-Process-Callback (jackio.py) kopiert nur zwischen Port-Puffern und den Ringen
(``AudioRing``, lock-frei, ohne Allokation). Die Engine liest in Blöcken von ``chunk`` Frames
(Vorgabe 2048 = 43 ms) aus den Quellringen, lässt ``RelayMixer.process`` samt RTT-Schätzung laufen
und schreibt das Programm in den Ausgangsring. Latenz im Programmweg ist egal (der Ausgang hat ohnehin
~3 s Puffer); xrun-Sicherheit geht vor.

Stereo: Der Mixer ist einkanalig. Der Master rechnet auf der Mitte M = (L+R)/2 (und misst darauf RTT),
ein Folger (``RelayMixer.follow``) auf der Seite S = (L−R)/2 mit derselben Umschaltliste, denselben
Lags und derselben Rate. Ausgabe: L = M+S, R = M−S - sample-genau phasenrichtig.

Alle Mixer-Aufrufe laufen im Worker-Thread. Andere Threads (HTTP, Router) reden mit der Engine über
eine Befehlsschlange (``call``) und lesen den zuletzt veröffentlichten Status (``snapshot``).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..mixer import RelayConfig, RelayMixer
from .config import LiveConfig
from .drift import DriftMonitor
from .ring import AudioRing
from .stats import lag_jitter_ms
from .wav import WavWriter

RING_FRAMES = 1 << 17           # 131072 Frames = 2,7 s je Quelle / Ausgang
OUT_PRIME_CHUNKS = 2            # Ausgang beginnt erst mit so vielen Blöcken Vorrat


class SourceInput:
    """Eingang einer Quelle (ein JackTrip-Client): Ring + Ports. ``origin`` setzt der Callback beim
    ersten Schreiben auf den globalen Frame-Index; ``enabled`` (Rolle dj) setzt die Engine."""

    def __init__(self, name: str, ports: Any = None) -> None:
        self.name = name
        self.ports = ports                  # (Port L, Port R), gehört jackio
        self.ring = AudioRing(2, RING_FRAMES)
        self.origin = -1
        self.o_eff = 0                      # origin + Frames, die der Überlauf verworfen hat
        self.drops_seen = 0
        self.enabled = False


@dataclass
class RtShared:
    """Vom Callback beschrieben, vom Worker gelesen (einzelne Attributzuweisungen, kein Lock)."""

    frames: int = 0                             # Frames, die der Callback vollständig verarbeitet hat
    stamp: tuple[int, float] = (0, 0.0)         # (Frame-Index am Zyklusanfang, CLOCK_MONOTONIC dort)
    xruns: int = 0
    cb_errors: int = 0
    out_underruns: int = 0
    cpu_load: float = 0.0


class Engine:
    def __init__(self, cfg: LiveConfig, mono: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.fs = cfg.sample_rate
        self.mono = mono
        self.rt = RtShared()
        self.inputs: tuple[SourceInput, ...] = ()       # tuple wird ausgetauscht, nie verändert
        self.program = AudioRing(2, RING_FRAMES)
        self.out_primed = False
        # Schätzung hinter dem JackTrip-Hub (S3904): Lags sind stückweise konstant und springen um ganze
        # JackTrip-Slots (Unter-/Überlauf, Verlust im Jitterpuffer) - keine Driftrate; Messungen bis ±12 ms
        # (> 2 Slots) um die Schätzung gehen in den Mittelwert über 12 s ein (= mittlerer Versatz der Musik),
        # gesucht wird ±20 ms, weiter weg = Ausreißer (Neu-Aufsetzen nach 2 s); Einrasten nur aus getrennten
        # Messfenstern, laufende Prüfung gegen den vollen Suchbereich; Aussetzer (exakte Nullen) verschleiert
        est_window_s, est_hop_s = 4.0, 0.5
        rc = RelayConfig(sample_rate=self.fs, buffer_s=cfg.target_buffer_s,
                         target_buffer_s=cfg.target_buffer_s, min_buffer_s=cfg.min_buffer_s,
                         max_dev=cfg.max_dev, fade_s=cfg.fade_s, bpm_ref=cfg.bpm_ref,
                         search_max_s=3.0, block_size=cfg.chunk,
                         est_window_s=est_window_s, est_hop_s=est_hop_s, max_rate_ppm=cfg.max_rate_ppm,
                         track_window_s=0.020, accept_s=0.012, fit_span_s=12.0,
                         confirm_s=est_window_s + est_hop_s, relock_s=2.0,
                         verify_every_s=2.0, reacquire_narrow_s=12.0,
                         conceal_max_s=cfg.conceal_max_ms * 1e-3)
        self.master = RelayMixer(rc)
        self.side = RelayMixer(rc)
        self.side.follow(self.master)
        self.drift = DriftMonitor(self.fs, enable=cfg.drift_enable,
                                  deadband_ms=cfg.drift_deadband_ms,
                                  slew_ms_per_s=cfg.drift_slew_ms_per_s)
        self._base = int(round(cfg.drift_base_s * self.fs))
        self._registered: list[str] = []
        self._cmds: queue.SimpleQueue = queue.SimpleQueue()
        self._snap: dict = {"started": False}
        self._snap_lock = threading.Lock()
        self._next_snap = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.proc = 0                                   # globaler Frame-Index, bis zu dem verarbeitet ist
        self._started_mono = 0.0
        self._last_active: str | None = None
        self._last_label = ""
        self._wav: WavWriter | None = None
        self._trace: Any = None                         # JSON-Lines neben dem Mitschnitt (Auswertung)
        self._trace_frames = 0                          # Frames, die in den Mitschnitt geschrieben sind
        self._retime_total = 0                          # Summe aller Umlabel-Schritte (Samples)
        self._last_event: Any = None
        self.load = 0.0                                 # Rechenzeit / Blockdauer (geglättet)
        self.chunks = 0
        self.last_error: str | None = None
        self._k = cfg.chunk
        self._l = {}
        self._r = {}

    # ------------------------------------------------------------------ Eingänge (andere Threads)

    def attach(self, si: SourceInput) -> None:
        self.inputs = self.inputs + (si,)

    def input(self, name: str) -> SourceInput | None:
        return next((i for i in self.inputs if i.name == name), None)

    def call(self, fn: Callable[[], Any], timeout: float = 5.0) -> Any:
        """``fn`` im Worker-Thread ausführen und das Ergebnis zurückgeben (oder deren Ausnahme)."""
        if self._thread is None or not self._thread.is_alive():
            return fn()
        fut: Future = Future()
        self._cmds.put((fn, fut))
        return fut.result(timeout)

    def snapshot(self) -> dict:
        with self._snap_lock:
            return self._snap

    # ------------------------------------------------------------------ Befehle (im Worker)

    def enable_source(self, name: str, enabled: bool) -> None:
        def run():
            si = self.input(name)
            if si is None:
                raise KeyError(name)
            if enabled and name not in self._registered:
                self.master.register_source(name)
                self.side.register_source(name)
                self._registered.append(name)
                if self.cfg.auto_active and self.master.status.active is None:
                    self._set_first(name)
            if not enabled and self.master.status.active == name:
                raise ValueError(f"{name} ist live; erst übergeben")
            si.enabled = enabled
        return self.call(run)

    def _set_first(self, name: str) -> None:
        self.master.set_active(name)
        self.master.retime(-self._base)
        self._retime_total += -self._base       # Vorlauf: alle Lags bleiben positiv (±drift_base_s Regelweg)
        self._last_active = name

    def handover(self, to: str) -> dict:
        def run():
            if to not in self._registered:
                return {"accepted": False, "reason": "unknown_source",
                        "message": f"{to!r} ist kein verbundener DJ"}
            if self.master.status.active is None:
                self._set_first(to)
                return {"accepted": True, "reason": "", "message": f"{to!r} ist jetzt live (erste Quelle)"}
            r = self.master.request_handoff(to)
            return {"accepted": r.accepted, "reason": r.reason, "message": r.message,
                    "buffer_s": r.buffer_s, "wait_s": r.wait_s}
        return self.call(run)

    # ------------------------------------------------------------------ Worker

    def start(self) -> None:
        self.proc = self.rt.frames
        self._started_mono = self.mono()
        self._open_record()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="b2b-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._wav:
            self._wav.close()
            self._wav = None
        if self._trace:
            self._trace.close()
            self._trace = None

    def _open_record(self) -> None:
        if not self.cfg.record_dir:
            return
        d = Path(self.cfg.record_dir)
        d.mkdir(parents=True, exist_ok=True)
        self._last_label = str(d / datetime.now().strftime("program-%Y%m%d-%H%M%S.wav"))
        self._wav = WavWriter(self._last_label, self.fs, 2, self.cfg.record_format)
        self._trace = open(self._last_label[:-4] + ".trace.jsonl", "w", buffering=1)
        self._trace_frames = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._drain_cmds()
                if not self.pump():
                    time.sleep(0.004)
            except Exception as e:          # Worker darf nie sterben: Fehler melden, weiterlaufen
                self.last_error = f"{type(e).__name__}: {e}"
                time.sleep(0.05)

    def _drain_cmds(self) -> None:
        while True:
            try:
                fn, fut = self._cmds.get_nowait()
            except queue.Empty:
                return
            try:
                fut.set_result(fn())
            except Exception as e:
                fut.set_exception(e)

    def pump(self) -> bool:
        """Einen Block verarbeiten, wenn genug Frames da sind. Gibt zurück, ob einer lief."""
        k = self._k
        if self.rt.frames - self.proc < k:
            return False
        t0 = time.perf_counter()
        self._chunk(k)
        dt = time.perf_counter() - t0
        self.load = 0.9 * self.load + 0.1 * dt / (k / self.fs)
        return True

    def _fetch(self, si: SourceInput, k: int, dl: np.ndarray, dr: np.ndarray) -> None:
        ring = si.ring
        if not si.enabled or si.origin < 0:
            ring.skip(ring.available)
            dl[:] = 0.0
            dr[:] = 0.0
            return
        if ring.dropped != si.drops_seen:            # Überlauf: Lücke, neu aufsetzen
            ring.skip(ring.available)
            si.o_eff = si.origin + ring.dropped
            si.drops_seen = ring.dropped
        else:
            si.o_eff = si.o_eff or si.origin
        a = si.o_eff + ring.consumed
        if a < self.proc:
            a += ring.skip(self.proc - a)
        pre = min(max(a - self.proc, 0), k)
        need = k - pre
        dl[:pre] = 0.0
        dr[:pre] = 0.0
        if need and ring.available >= need:
            ring.read((dl[pre:], dr[pre:]), need)
        else:
            dl[pre:] = 0.0
            dr[pre:] = 0.0

    def _mono_at(self, g: int) -> float:
        f0, m0 = self.rt.stamp
        return m0 + (g - f0) / self.fs

    def _chunk(self, k: int) -> None:
        mid, sid = {}, {}
        for name in self._registered:
            si = self.input(name)
            dl = self._l.setdefault(name, np.zeros(k, np.float32))
            dr = self._r.setdefault(name, np.zeros(k, np.float32))
            self._fetch(si, k, dl, dr)
            mid[name] = (dl + dr) * 0.5
            sid[name] = (dl - dr) * 0.5
        m = self.master.process(mid, k)
        s = self.side.process(sid, k)
        left, right = m + s, m - s
        if self._wav:
            self._wav.write((left, right))
            if self._trace:
                self._write_trace(k)
        self.program.write((left, right), k)
        self.proc += k
        self.chunks += 1
        self._drift_step(k)
        now = self.mono()
        if now >= self._next_snap:
            self._next_snap = now + 0.25
            self._publish()

    def _write_trace(self, k: int) -> None:
        """Eine Zeile je Block: Mitschnitt-Index → Programmzeit (ohne Umlabeln, "Inhaltszeit"), Rate, Puffer;
        dazu neue Mixer-Ereignisse. Grundlage für ``tests/integration/analyze.py``."""
        m = self.master
        lb = m.last_block
        if lb is not None:
            rec = {"k": "blk", "i": self._trace_frames, "n": k, "Tc": lb.pos - self._retime_total,
                   "r0": round(lb.r0, 6), "r1": round(lb.r1, 6), "b": round(lb.buffer_s, 4),
                   "wall": m.wall, "drift_ms": round(self.drift.status().drift_ms, 3),
                   "xr": self.rt.xruns, "cc": sum(c for c, _ in m.concealed.values())}
            self._trace.write(json.dumps(rec) + "\n")
        evs = list(m.events)
        new = []
        for e in reversed(evs):
            if e is self._last_event:
                break
            new.append(e)
        if evs:
            self._last_event = evs[-1]
        for e in reversed(new):
            rec = {"k": "ev", "i": self._trace_frames, "wall": e.wall, "kind": e.kind, **e.detail}
            if "t_switch" in rec:
                rec["Tc_switch"] = rec["t_switch"] - self._retime_total
            self._trace.write(json.dumps(rec, default=str) + "\n")
        self._trace_frames += k

    def _drift_step(self, k: int) -> None:
        st = self.master.status
        a = st.active
        if a is None:
            return
        mono = self._mono_at(self.proc)
        if a != self._last_active:
            if self._last_active is not None and self.drift.anchored:
                self.drift.credit_handover(self.master.lag_of(a), self.master.lag_of(self._last_active))
            self._last_active = a
        if not self.drift.anchored:
            self.drift.anchor(self.master.live_edge, mono)
            return
        d = self.drift.step(self.master.live_edge, mono, k / self.fs)
        if d:
            self.master.retime(d)
            self._retime_total += d

    # ------------------------------------------------------------------ Status

    def _publish(self) -> None:
        m = self.master
        st = m.status
        bs = st.buffer
        sources = {}
        for name, si in st.sources.items():
            tr = m._sources[name].tracker
            sources[name] = {
                "locked": si.locked, "rtt_ms": None if si.rtt_s is None else round(si.rtt_s * 1e3, 2),
                "jitter_ms": _r(lag_jitter_ms(tr, self.fs), 2),
                "drift_ppm": _r(si.drift_ppm, 1), "confidence": round(si.confidence, 2),
                "level_db": round(si.level_db, 1), "silent": si.silent,
                "stale_s": _r(si.stale_s, 1)}
        d = self.drift.status()
        rt = self.rt
        snap = {
            "started": True,
            "uptime_s": round(self.mono() - self._started_mono, 1),
            "live": st.active, "previous": st.previous, "fallback": st.fallback,
            "silent": st.silent, "no_source": st.no_source,
            "buffer": None if bs is None else {
                "buffer_s": round(bs.buffer_s, 3), "target_s": bs.target_s, "min_s": bs.min_s,
                "rate": round(bs.rate, 5), "rate_bpm": round(bs.rate_bpm, 2),
                "handover_release": bs.release, "handover_wait_s": bs.wait_s,
                "handoff": {k: {"ok": o.ok, "buffer_after_s": round(o.buffer_after_s, 3),
                                "lag_growth_s": round(o.lag_growth_s, 3), "wait_s": o.wait_s}
                            for k, o in bs.handoff.items()}},
            "sources": sources,
            "drift": {"anchored": d.anchored, "drift_ms": round(d.drift_ms, 3),
                      "raw_drift_ms": round(d.raw_drift_ms, 3), "corrected_ms": round(d.corrected_ms, 3),
                      "ppm_60s": _r(d.ppm_60s, 1), "max_abs_ms": round(d.max_abs_ms, 3),
                      "credits_ms": round(d.credits_ms, 1)},
            "underruns": st.underruns,
            "jack": {"xruns": rt.xruns, "cb_errors": rt.cb_errors, "out_underruns": rt.out_underruns,
                     "cpu_load": round(rt.cpu_load, 1), "frames": rt.frames},
            "worker": {"load": round(self.load, 3), "chunks": self.chunks, "error": self.last_error,
                       "in_dropped": {i.name: i.ring.dropped for i in self.inputs},
                       "out_dropped": self.program.dropped},
            "recording": self._last_label if self._wav else None,
            "events": [{"t_s": round(e.wall / self.fs, 2), "kind": e.kind, **e.detail}
                       for e in list(m.events)[-20:]],
        }
        with self._snap_lock:
            self._snap = snap


def _r(x, nd):
    return None if x is None else round(float(x), nd)
