"""RelayMixer: blockweiser Relay-Mixer mit Programm-Zeitachse, RTT-Messung und Übergabe.

Ablauf je ``process``-Aufruf (alle Quellen liefern gleich lange Blöcke, die "jetzt" ankommen):

1. Blöcke in die Ringe der Quellen schreiben (fehlende Quelle = Nullen).
2. Aktive Quelle nachziehen (Programmzeit am Live-Rand hat den Umschaltpunkt passiert?).
3. Alle ``est_hop_s`` die RTT jeder wartenden Quelle gegen das Programm schätzen.
4. Fallback prüfen (aktive Quelle still → vorige Quelle).
5. Den Ausgabeblock mit festem Puffer ``D`` erzeugen: P(T) für T = w − D.

Das Programm P(T) wird nie als Block gespeichert, sondern beim Lesen aus den Quell-Ringen
zusammengesetzt: P(T) = Σ_k gewicht_k(T) · feed_k(T + lag_k). Die Gewichte ergeben sich aus der
Liste der Umschaltpunkte (``_Segment``). Dadurch wirken Umschaltpunkte und verbesserte
Lag-Schätzungen auch rückwirkend innerhalb des Puffers D.

Lag-Konvention: ``lag_X`` ist der Versatz in Samples, mit dem der Feed von X dem Programm
hinterherläuft. Die erste aktive Quelle definiert die Zeitachse (lag 0). Eine Quelle, die das
Programm als Rückkanal hört, kommt um den Lag der Programmquelle plus ihre eigene RTT zurück —
**der Lag summiert sich daher über Übergaben auf** (siehe README, "Grenzen").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rtt import LagTracker, gcc_phat
from .timeline import SampleRing

_BIG = 1 << 60


@dataclass
class RelayConfig:
    sample_rate: int = 48000
    buffer_s: float = 2.0           # D: fester Ausgabepuffer
    reserve_s: float = 0.3          # Jitter-/Schätzreserve: Übergabe nur wenn lag + reserve ≤ D
    fade_s: float = 0.1             # Crossfade bei Übergabe (50–200 ms)
    block_size: int = 512           # Vorgabe, wenn process() ohne Blöcke aufgerufen wird
    est_window_s: float = 4.0
    est_min_window_s: float = 2.0
    est_hop_s: float = 0.5
    search_max_s: float | None = None   # größter gesuchter Lag, Vorgabe 1.5·D
    min_rtt_s: float = 0.004
    min_conf: float = 0.5
    gate_s: float = 0.003
    max_rate_ppm: float = 500.0
    silence_db: float = -60.0
    silence_timeout_s: float = 0.25     # Stille der aktiven Quelle, ab der der Fallback greift
    rearm_silence_s: float = 1.0        # Stille, nach der eine Quelle als "neu angeschlossen" gilt
    max_stale_s: float = 8.0            # ältere Schätzung → Übergabe verweigert
    ring_s: float | None = None

    def __post_init__(self) -> None:
        if not 0.05 <= self.fade_s <= 0.2:
            raise ValueError("fade_s muss zwischen 0.05 und 0.2 s liegen")
        if self.buffer_s <= self.reserve_s:
            raise ValueError("buffer_s muss größer als reserve_s sein")
        if self.search_max_s is None:
            self.search_max_s = 1.5 * self.buffer_s
        if self.ring_s is None:
            self.ring_s = self.est_window_s + self.search_max_s + self.buffer_s + 2.0


@dataclass(frozen=True)
class HandoffResult:
    accepted: bool
    reason: str                 # "ok" oder Code der Verweigerung
    message: str
    target: str
    t_switch: int | None = None     # Umschaltpunkt in Programm-Samples
    lag_s: float | None = None      # Lag des Ziels relativ zum Programm (Schätzung oder Hinweis)
    required_s: float | None = None  # lag + reserve
    buffer_s: float | None = None


@dataclass(frozen=True)
class Event:
    wall: int
    kind: str
    detail: dict


@dataclass(frozen=True)
class SourceInfo:
    name: str
    locked: bool
    lag_s: float | None
    rtt_s: float | None             # Lag relativ zur Programmquelle
    drift_ppm: float | None
    confidence: float
    level_db: float
    silent: bool
    stale_s: float | None


@dataclass(frozen=True)
class MixerStatus:
    active: str | None
    previous: str | None
    no_source: bool             # keine Quelle aktiv → Ausgabe ist Stille
    silent: bool                # aktive Quelle liefert Stille, kein Fallback möglich
    fallback: bool              # letzte Umschaltung war ein Fallback
    underruns: int              # Ausgabeblöcke mit noch nicht angekommenen Quelldaten
    wall_s: float
    sources: dict[str, SourceInfo] = field(default_factory=dict)


@dataclass
class _Segment:
    t_start: int
    src: str
    fade: int


class _Source:
    def __init__(self, name: str, ring: SampleRing, tracker: LagTracker, rtt_hint_s: float | None):
        self.name = name
        self.ring = ring
        self.tracker = tracker
        self.rtt_hint_s = rtt_hint_s
        self.silence_run = _BIG     # Samples Stille am Ende des bisherigen Feeds
        self.signal_start: int | None = None
        self.had_signal = False
        self.level_db = -120.0
        self.last_active_wall: int | None = None
        self.is_reference = False


class RelayMixer:
    def __init__(self, config: RelayConfig | None = None) -> None:
        self.cfg = config or RelayConfig()
        c = self.cfg
        self.fs = c.sample_rate
        self._D = int(round(c.buffer_s * self.fs))
        self._fade = int(round(c.fade_s * self.fs))
        self._sources: dict[str, _Source] = {}
        self._segments: list[_Segment] = []
        self._wall = 0
        self._active: str | None = None
        self._previous: str | None = None
        self._fallback = False
        self._silent_flag = False
        self._underruns = 0
        self._next_est = int(c.est_hop_s * self.fs)
        self._sil_thr = 10.0 ** (c.silence_db / 20.0)
        self._sil_timeout = int(c.silence_timeout_s * self.fs)
        self._rearm = int(c.rearm_silence_s * self.fs)
        self.events: list[Event] = []

    # ------------------------------------------------------------------ Quellen

    def register_source(self, name: str, rtt_hint_s: float | None = None) -> None:
        """Quelle anmelden. ``rtt_hint_s`` (z. B. aus einem Ping) dient nur der D-Prüfung,
        solange die Audio-Schätzung noch nicht eingerastet ist."""
        if name in self._sources:
            raise ValueError(f"Quelle {name!r} ist schon registriert")
        c = self.cfg
        tracker = LagTracker(gate=c.gate_s * self.fs, rate_limit=c.max_rate_ppm * 1e-6,
                             min_conf=c.min_conf)
        self._sources[name] = _Source(name, SampleRing(int(c.ring_s * self.fs)), tracker, rtt_hint_s)

    def set_active(self, name: str) -> None:
        """Erste Quelle aktivieren: sie definiert die Zeitachse (lag 0)."""
        s = self._get(name)
        if self._segments:
            raise RuntimeError("set_active ist nur für die erste Quelle; danach request_handoff")
        s.tracker.set_reference(0.0, float(self._wall))
        s.is_reference = True
        self._segments.append(_Segment(self._wall, name, 1))
        self._active = name
        self._event("active_set", source=name)

    # ------------------------------------------------------------------ Übergabe

    def check_handoff(self, target: str, at: float | None = None) -> HandoffResult:
        """Wie ``request_handoff``, ändert aber nichts."""
        return self._handoff(target, at, commit=False)

    def request_handoff(self, target: str, at: float | None = None) -> HandoffResult:
        """Übergabe an ``target``: sofort (``at=None``, am Live-Rand des Programms) oder zur
        Programmzeit ``at`` (Sekunden auf der Zeitachse T). Verweigert mit Grund, wenn der
        Lag des Ziels plus Reserve den Puffer D sprengt, die Schätzung fehlt/veraltet ist oder
        der Zeitpunkt nicht mehr im Puffer liegt."""
        return self._handoff(target, at, commit=True)

    def _handoff(self, target: str, at: float | None, commit: bool) -> HandoffResult:
        c = self.cfg

        def no(reason: str, msg: str, **kw) -> HandoffResult:
            if commit:
                self._event("handoff_refused", target=target, reason=reason)
            return HandoffResult(False, reason, msg, target, buffer_s=c.buffer_s, **kw)

        if target not in self._sources:
            return no("unknown_source", f"Quelle {target!r} ist nicht registriert")
        if not self._segments:
            return no("no_active_source", "Es gibt noch keine aktive Quelle (set_active)")
        s = self._sources[target]
        if self._segments[-1].src == target:
            return no("already_active", f"{target!r} ist schon aktiv oder bereits vorgemerkt")
        act_lag = self.lag_of(self._active) if self._active else 0.0
        # Lag des Ziels relativ zur Zeitachse: Schätzung, sonst Programmquelle + RTT-Hinweis
        if s.tracker.locked:
            lag_s = s.tracker.predict(self._wall) / self.fs
            src = "Schätzung"
        elif s.rtt_hint_s is not None:
            lag_s = act_lag / self.fs + s.rtt_hint_s
            src = "RTT-Hinweis"
        else:
            lag_s = None
            src = ""
        if lag_s is not None and lag_s + c.reserve_s > c.buffer_s:
            return no("rtt_exceeds_buffer",
                      f"Lag von {target!r} ({lag_s * 1e3:.0f} ms, {src}) plus Reserve "
                      f"({c.reserve_s * 1e3:.0f} ms) übersteigt den Puffer D = {c.buffer_s * 1e3:.0f} ms",
                      lag_s=lag_s, required_s=lag_s + c.reserve_s)
        if not s.tracker.locked:
            return no("not_locked", f"Für {target!r} gibt es noch keine sichere RTT-Schätzung",
                      lag_s=lag_s)
        stale = None if s.tracker.last_update is None else (self._wall - s.tracker.last_update) / self.fs
        if stale is not None and stale > c.max_stale_s and not s.is_reference:
            return no("estimate_stale", f"Letzte gute Messung für {target!r} ist {stale:.1f} s alt",
                      lag_s=lag_s)
        if s.silence_run >= self._sil_timeout:
            return no("target_silent", f"{target!r} liefert gerade Stille", lag_s=lag_s)
        t_next_out = self._wall - self._D
        if at is None:
            t_sw = int(round(self._wall - act_lag))      # Live-Rand des Programms
        else:
            t_sw = int(round(at * self.fs))
        if t_sw < t_next_out:
            return no("too_late", "Der Umschaltpunkt liegt schon vor dem Ausgabe-Rand "
                      f"({(t_next_out - t_sw) / self.fs:.2f} s zu spät)", lag_s=lag_s)
        if t_sw < self._segments[-1].t_start:
            return no("out_of_order", "Es ist bereits eine spätere Umschaltung vorgemerkt", lag_s=lag_s)
        if commit:
            self._segments.append(_Segment(t_sw, target, self._fade))
            self._fallback = False
            self._event("handoff_scheduled", target=target, t_switch=t_sw, lag_s=lag_s)
            self._update_active()
        return HandoffResult(True, "ok", "Übergabe vorgemerkt" if commit else "Übergabe möglich",
                             target, t_switch=t_sw, lag_s=lag_s, required_s=lag_s + c.reserve_s,
                             buffer_s=c.buffer_s)

    # ------------------------------------------------------------------ Verarbeitung

    def process(self, blocks_by_source: dict[str, np.ndarray], n: int | None = None) -> np.ndarray:
        """Einen Block verarbeiten; gibt P(T) für T = w − D zurück (Stille, solange T < 0)."""
        for name in blocks_by_source:
            if name not in self._sources:
                raise ValueError(f"unbekannte Quelle {name!r}")
        if n is None:
            n = len(next(iter(blocks_by_source.values()))) if blocks_by_source else self.cfg.block_size
        w0 = self._wall
        for name, s in self._sources.items():
            blk = blocks_by_source.get(name)
            if blk is None:
                blk = np.zeros(n, dtype=np.float32)
            elif len(blk) != n:
                raise ValueError("alle Blöcke müssen gleich lang sein")
            else:
                blk = np.asarray(blk, dtype=np.float32)
            self._track_level(s, blk, w0)
            s.ring.write(blk)
        self._wall = w0 + n
        self._update_active()
        if self._wall >= self._next_est:
            self._run_estimators()
            self._next_est = self._wall + int(self.cfg.est_hop_s * self.fs)
        self._check_fallback(n)
        out = self._compose(w0 - self._D, n, count=True).astype(np.float32)
        self._prune()
        return out

    def read_program(self, t0: int, n: int) -> np.ndarray:
        """Programm P(T) für T in [t0, t0+n) mit dem heutigen Stand (Rückkanal/Monitor).
        Was noch nicht angekommen ist, kommt als Null."""
        return self._compose(int(t0), int(n), count=False)

    # ------------------------------------------------------------------ Auskunft

    @property
    def wall(self) -> int:
        return self._wall

    @property
    def live_edge(self) -> int:
        """Jüngste Programmzeit, für die die aktive Quelle schon Daten geliefert hat."""
        return int(round(self._wall - (self.lag_of(self._active) if self._active else 0.0)))

    @property
    def output_time(self) -> int:
        """Programmzeit des nächsten Ausgabe-Samples (= w − D)."""
        return self._wall - self._D

    def lag_of(self, name: str, wall: float | None = None) -> float:
        """Lag der Quelle in Samples (Schätzung, mit Drift fortgeschrieben)."""
        return self._get(name).tracker.predict(self._wall if wall is None else wall)

    def source_info(self, name: str) -> SourceInfo:
        s = self._get(name)
        tr = s.tracker
        ref = self._active if (self._active and self._active != name) else self._previous
        rtt = None
        if tr.locked and ref is not None and ref != name:
            rtt = (tr.predict(self._wall) - self.lag_of(ref)) / self.fs
        return SourceInfo(
            name=name, locked=tr.locked,
            lag_s=tr.predict(self._wall) / self.fs if tr.locked else None,
            rtt_s=rtt, drift_ppm=tr.rate * 1e6 if tr.locked else None,
            confidence=tr.confidence, level_db=s.level_db,
            silent=s.silence_run >= self._sil_timeout,
            stale_s=None if tr.last_update is None else (self._wall - tr.last_update) / self.fs)

    @property
    def status(self) -> MixerStatus:
        silent = bool(self._active and self._silent_flag)
        return MixerStatus(
            active=self._active, previous=self._previous, no_source=self._active is None,
            silent=silent, fallback=self._fallback, underruns=self._underruns,
            wall_s=self._wall / self.fs,
            sources={k: self.source_info(k) for k in self._sources})

    # ------------------------------------------------------------------ intern

    def _get(self, name: str) -> _Source:
        try:
            return self._sources[name]
        except KeyError:
            raise KeyError(f"Quelle {name!r} ist nicht registriert") from None

    def _event(self, kind: str, **detail) -> None:
        self.events.append(Event(self._wall, kind, detail))

    def _track_level(self, s: _Source, blk: np.ndarray, w0: int) -> None:
        n = len(blk)
        quiet = np.abs(blk) < self._sil_thr
        if quiet.all():
            s.silence_run = min(s.silence_run + n, _BIG)
            s.level_db = -120.0
            return
        loud = np.flatnonzero(~quiet)
        if s.silence_run >= self._rearm:
            # Quelle (wieder) angeschlossen → Messung frisch beginnen
            s.signal_start = w0 + int(loud[0])
            if s.name != self._active and not s.is_reference:
                s.tracker.reset()
        s.silence_run = n - 1 - int(loud[-1])
        s.had_signal = True
        s.level_db = 20.0 * np.log10(float(np.sqrt(np.mean(np.square(blk, dtype=np.float64)))) + 1e-12)

    def _lag_params(self, s: _Source) -> tuple[float, float, float]:
        return s.tracker.lag, s.tracker.rate, s.tracker.t_ref

    def _positions(self, s: _Source, T: np.ndarray) -> np.ndarray:
        """Wanduhr-Leseposition w mit w = T + lag(w) (lag linear in w)."""
        l0, rho, tref = self._lag_params(s)
        return (T + l0 - rho * tref) / (1.0 - rho)

    def _weights(self, t0: int, n: int) -> dict[str, np.ndarray | float]:
        weights: dict[str, np.ndarray | float] = {}
        T = None
        for seg in self._segments:
            if seg.t_start >= t0 + n:
                break
            if t0 >= seg.t_start + seg.fade:
                weights = {seg.src: 1.0}
                continue
            if T is None:
                T = np.arange(t0, t0 + n, dtype=np.float64)
            x = np.clip((T - seg.t_start) / seg.fade, 0.0, 1.0)
            r = 0.5 - 0.5 * np.cos(np.pi * x)       # Summe der Gewichte bleibt 1
            for k in list(weights):
                weights[k] = weights[k] * (1.0 - r)
            weights[seg.src] = weights.get(seg.src, 0.0) * (1.0 - r) + r
        return weights

    def _compose(self, t0: int, n: int, count: bool) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        weights = self._weights(t0, n)
        if not weights:
            return out
        T = np.arange(t0, t0 + n, dtype=np.float64)
        for name, wgt in weights.items():
            if isinstance(wgt, float) and wgt == 0.0:
                continue
            if not isinstance(wgt, float) and not np.any(wgt):
                continue
            s = self._sources[name]
            pos = self._positions(s, T)
            if count and pos[-1] + 2 > s.ring.end:
                self._underruns += 1
            out += wgt * s.ring.read_frac(pos)
        return out

    def _update_active(self) -> None:
        if not self._segments or self._active is None:
            return
        for _ in range(8):
            t_raw = self._wall - self.lag_of(self._active)
            cand = None
            for seg in self._segments:
                if seg.t_start <= t_raw:
                    cand = seg
            if cand is None or cand.src == self._active:
                return
            old = self._active
            self._sources[old].last_active_wall = self._wall
            self._previous, self._active = old, cand.src
            self._silent_flag = False
            self._event("handoff_effective", source=cand.src, previous=old, t_switch=cand.t_start)

    def _prune(self) -> None:
        horizon = self._wall - self._D - int(self.cfg.ring_s * self.fs)
        while len(self._segments) > 1 and \
                self._segments[1].t_start + self._segments[1].fade < horizon:
            self._segments.pop(0)

    # -- Schätzung

    def _run_estimators(self) -> None:
        if self._active is None:
            return
        c = self.cfg
        for s in self._sources.values():
            if s.name == self._active or s.is_reference or s.signal_start is None:
                continue
            if s.silence_run >= int(0.5 * self.fs):
                continue
            if s.last_active_wall is not None and \
                    self._wall - s.last_active_wall < int((c.est_window_s + 1.0) * self.fs):
                continue
            if self._segments[-1].src == s.name:
                continue    # Übergabe an s ist vorgemerkt oder läuft: Schätzung eingefroren
            self._estimate(s)

    def _estimate(self, s: _Source) -> None:
        c = self.cfg
        fs = self.fs
        w_end = self._wall
        n_win = min(int(c.est_window_s * fs), w_end - s.signal_start)
        if n_win < int(c.est_min_window_s * fs):
            return
        x = s.ring.read(w_end - n_win, n_win)
        if float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) < self._sil_thr:
            return
        w0 = w_end - n_win
        t_c = w0 + n_win / 2.0
        act_lag = self.lag_of(self._active)
        if s.tracker.locked:
            pred = s.tracker.predict(t_c)
            lo = int(np.floor(pred - s.tracker.gate))
            hi = int(np.ceil(pred + s.tracker.gate))
        else:
            lo = int(np.ceil(act_lag + c.min_rtt_s * fs))
            hi = int(c.search_max_s * fs)
        lo = max(lo, 1)
        if hi - lo < 16:
            return
        p = self._compose(w0 - hi, n_win + (hi - lo), count=False).astype(np.float32)
        if float(np.sqrt(np.mean(np.square(p, dtype=np.float64)))) < self._sil_thr:
            return
        m = gcc_phat(x, p, hi - lo)
        if m is None:
            return
        d = lo + m.delay    # gcc_phat misst relativ zur unteren Suchgrenze lo
        if d <= lo + 1.5 or d >= hi - 1.5:
            if not s.tracker.locked:
                return
        s.tracker.update(t_c, d, m.confidence)

    # -- Fallback

    def _check_fallback(self, n: int) -> None:
        a = self._active
        if a is None:
            return
        s = self._sources[a]
        if not s.had_signal or s.silence_run < self._sil_timeout:
            self._silent_flag = False
            return
        prev = self._sources.get(self._previous) if self._previous else None
        if prev is None or prev.name == a or not prev.tracker.locked \
                or prev.silence_run >= self._sil_timeout:
            if not self._silent_flag:
                self._event("silence_no_fallback", source=a)
            self._silent_flag = True
            return
        w_sil = self._wall - s.silence_run
        t_sil = w_sil - self.lag_of(a, w_sil)
        t_sw = max(int(t_sil), self._wall - n - self._D, self._segments[-1].t_start)
        self._segments.append(_Segment(t_sw, prev.name, self._fade))
        self._fallback = True
        self._silent_flag = False
        self._event("fallback", source=a, to=prev.name, t_switch=t_sw)
        self._update_active()
