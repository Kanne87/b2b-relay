"""RelayMixer: blockweiser Relay-Mixer mit Programm-Zeitachse, RTT-Messung und Übergabe.

Ablauf je ``process``-Aufruf (alle Quellen liefern gleich lange Blöcke, die "jetzt" ankommen):

1. Blöcke in die Ringe der Quellen schreiben (fehlende Quelle = Nullen).
2. Aktive Quelle nachziehen (Programmzeit am Live-Rand hat den Umschaltpunkt passiert?).
3. Alle ``est_hop_s`` die RTT jeder wartenden Quelle gegen das Programm schätzen.
4. Fallback prüfen (aktive Quelle still → vorige Quelle).
5. Den Ausgabeblock erzeugen. Ohne Regelung (``target_buffer_s=None``) mit festem Puffer ``D``:
   P(T) für T = w − D. Mit Regelung liest eine Varispeed-Stufe P(T) mit Rate ``r`` (siehe
   ``control.py``, ``varispeed.py``); der Puffer ``B = Live-Rand − Ausgabeposition`` wird auf
   ``target_buffer_s`` geregelt, statt mit jeder Übergabe um den Lag-Zuwachs zu schrumpfen.

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

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .control import BufferController
from .rtt import LagTracker, gcc_phat
from .timeline import SampleRing
from .varispeed import VarispeedReader

_BIG = 1 << 60


@dataclass
class RelayConfig:
    sample_rate: int = 48000
    buffer_s: float = 2.0           # D: Ausgabepuffer (bei Regelung: Anfangswert)
    reserve_s: float = 0.3          # Jitter-/Schätzreserve: Übergabe nur wenn lag + reserve ≤ D
                                    # (nur ohne Regelung; mit Regelung gilt min_buffer_s)
    # --- elastischer Ausgang (None = fester Puffer D wie in Scheibe 1, Rate immer 1)
    target_buffer_s: float | None = None    # Zielpuffer B_ziel (Empfehlung 3,0 s)
    min_buffer_s: float = 1.0               # B_min: Untergrenze nach einer Übergabe
    max_dev: float = 0.005                  # |r − 1| höchstens (0,5 %)
    max_slew: float = 0.001                 # größte Änderung von r je Sekunde
    deadband_s: float = 0.1                 # Totband ±, in dem r = 1 gehalten wird
    bpm_ref: float = 174.0                  # nur für die Anzeige „äquivalente BPM“
    fade_s: float = 0.1             # Crossfade bei Übergabe (50–200 ms)
    block_size: int = 512           # Vorgabe, wenn process() ohne Blöcke aufgerufen wird
    est_window_s: float = 4.0
    est_min_window_s: float = 2.0
    est_hop_s: float = 0.5
    search_max_s: float | None = None   # größte gesuchte RTT über dem Lag der aktiven Quelle, Vorgabe 1.5·D
    min_rtt_s: float = 0.004
    min_conf: float = 0.5
    gate_s: float = 0.003
    max_rate_ppm: float = 500.0
    silence_db: float = -60.0
    silence_timeout_s: float = 0.25     # Stille der aktiven Quelle, ab der der Fallback greift
    rearm_silence_s: float = 1.0        # Stille, nach der eine Quelle als "neu angeschlossen" gilt
    max_stale_s: float = 8.0            # ältere Schätzung → Übergabe verweigert
    reacquire_delay_s: float = 3.0      # nach einem Wechsel der aktiven Quelle: Messfenster der übrigen erst
                                        # ab hier (Rückkanal umgeschaltet, neuer Feed angekommen, Ausklang vorbei)
    ring_s: float | None = None

    def __post_init__(self) -> None:
        if not 0.05 <= self.fade_s <= 0.2:
            raise ValueError("fade_s muss zwischen 0.05 und 0.2 s liegen")
        if self.buffer_s <= self.reserve_s:
            raise ValueError("buffer_s muss größer als reserve_s sein")
        if self.search_max_s is None:
            self.search_max_s = 1.5 * self.buffer_s
        if self.target_buffer_s is not None:
            if not 0.0 < self.min_buffer_s < self.target_buffer_s:
                raise ValueError("min_buffer_s muss zwischen 0 und target_buffer_s liegen")
            if self.target_buffer_s - self.deadband_s <= self.min_buffer_s:
                raise ValueError("Totband reicht bis an min_buffer_s heran")
            if self.buffer_s < self.min_buffer_s:
                raise ValueError("buffer_s (Anfangspuffer) liegt unter min_buffer_s")
            if not 0.0 < self.max_dev <= 0.05 or self.max_slew <= 0.0 or self.deadband_s < 0.0:
                raise ValueError("max_dev (0..0,05], max_slew > 0 und deadband_s >= 0 verlangt")
        if self.ring_s is None:
            # Der Ring hält nur das Fenster Ausgabe..Ankunft (B plus RTT der Übergabe) und das
            # Schätzfenster; der absolute Lag (wächst über Stunden) geht nicht ein.
            b_max = max(self.buffer_s, self.target_buffer_s or 0.0) + self.deadband_s
            self.ring_s = self.est_window_s + self.search_max_s + b_max + 2.0


@dataclass(frozen=True)
class HandoffResult:
    accepted: bool
    reason: str                 # "ok" oder Code der Verweigerung
    message: str
    target: str
    t_switch: int | None = None     # Umschaltpunkt in Programm-Samples
    lag_s: float | None = None      # Lag des Ziels relativ zum Programm (Schätzung oder Hinweis)
    required_s: float | None = None  # lag + reserve (ohne Regelung)
    buffer_s: float | None = None    # D bzw. aktueller Puffer B
    buffer_after_s: float | None = None   # B nach der Übergabe (nur mit Regelung)
    wait_s: float | None = None      # geschätzte Wartezeit bis zur Freigabe (nur mit Regelung)


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
class HandoffOutlook:
    """Freigabe der Übergabe an eine Quelle nach dem Pufferkriterium."""

    ok: bool
    buffer_after_s: float           # erwarteter Puffer nach der Übergabe
    lag_growth_s: float             # erwarteter Lag-Zuwachs
    wait_s: float | None            # 0 = frei; None = mit den Reglerwerten nicht erreichbar


@dataclass(frozen=True)
class BufferStatus:
    """Zustand des elastischen Ausgangs (Vorbereitung für die Oberfläche)."""

    enabled: bool
    buffer_s: float                 # B: angekommen, aber noch nicht ausgegeben
    target_s: float | None
    min_s: float | None
    rate: float                     # r (Ausgabe-Tempo; < 1 = langsamer, füllt auf)
    rate_bpm: float                 # äquivalente BPM (bpm_ref · r)
    output_delay_s: float           # D_eff = w − T_aus (wächst mit der Summe der Lags)
    lag_s: float                    # Lag der aktiven Quelle
    release: bool                   # Übergabe an irgendeine Quelle nach B-Kriterium frei
    wait_s: float | None            # kürzeste Wartezeit bis zur Freigabe (0 = frei)
    eta_target_s: float | None      # Sekunden, bis B im Totband um das Ziel liegt
    handoff: dict[str, HandoffOutlook] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockInfo:
    """Leseposition des zuletzt verarbeiteten Ausgabeblocks (Position in Programm-Samples)."""

    n: int
    pos: float                      # Programmzeit des ersten Ausgabe-Samples
    r0: float                       # Rate am Blockanfang
    r1: float                       # Rate am Blockende (linearer Verlauf)
    buffer_s: float                 # B am Blockende


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
    buffer: BufferStatus | None = None


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
        self.est_from: int | None = None   # Wanduhr, ab der das Messfenster beginnen darf (Neu-Einrasten)


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
        self._cur_seg: _Segment | None = None
        self._previous: str | None = None
        self._fallback = False
        self._silent_flag = False
        self._underruns = 0
        self._next_est = int(c.est_hop_s * self.fs)
        self._sil_thr = 10.0 ** (c.silence_db / 20.0)
        self._sil_timeout = int(c.silence_timeout_s * self.fs)
        self._rearm = int(c.rearm_silence_s * self.fs)
        self.events: deque[Event] = deque(maxlen=4096)    # begrenzt: läuft über Stunden
        self._t_blk = -self._D          # Programmzeit des Ausgabeblocks, der gerade entsteht
        self.last_block: BlockInfo | None = None
        self._stage: VarispeedReader | None = None
        self._ctl: BufferController | None = None
        self._master: RelayMixer | None = None      # Folger-Modus (siehe ``follow``)
        self._blk_start: tuple[int, float] = (0, 0.0)   # Leseposition (int, frac) vor dem Block
        if c.target_buffer_s is not None:
            self._stage = VarispeedReader(-float(self._D))
            self._ctl = BufferController(c.target_buffer_s, c.deadband_s, c.max_dev, c.max_slew)

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

    def set_lag(self, name: str, lag_s: float) -> None:
        """Lag einer Quelle von außen vorgeben (z. B. aus einer anderen Messung; ohne Audio-
        Schätzung). Die Quelle gilt dann als eingerastet, mit Driftrate 0."""
        s = self._get(name)
        if s.is_reference and name == self._active:
            raise ValueError("Lag der aktiven Quelle ist durch die Zeitachse festgelegt")
        s.tracker.set_reference(lag_s * self.fs, float(self._wall))

    def set_active(self, name: str) -> None:
        """Erste Quelle aktivieren: sie definiert die Zeitachse (lag 0)."""
        s = self._get(name)
        if self._segments:
            raise RuntimeError("set_active ist nur für die erste Quelle; danach request_handoff")
        s.tracker.set_reference(0.0, float(self._wall))
        s.is_reference = True
        self._cur_seg = _Segment(self._wall - 1, name, 1)
        self._segments.append(self._cur_seg)
        self._active = name
        self._event("active_set", source=name)

    # ------------------------------------------------------------------ Übergabe

    def check_handoff(self, target: str, at: float | None = None) -> HandoffResult:
        """Wie ``request_handoff``, ändert aber nichts."""
        return self._handoff(target, at, commit=False)

    def request_handoff(self, target: str, at: float | None = None) -> HandoffResult:
        """Übergabe an ``target``: sofort (``at=None``, am Live-Rand des Programms) oder zur
        Programmzeit ``at`` (Sekunden auf der Zeitachse T). Verweigert mit Grund, wenn der
        Puffer nach dem erwarteten Lag-Zuwachs unter ``min_buffer_s`` fiele (mit Regelung; die
        Meldung nennt die geschätzte Wartezeit) bzw. der Lag des Ziels plus Reserve den festen
        Puffer D sprengt (ohne Regelung), die Schätzung fehlt/veraltet ist oder der Zeitpunkt
        nicht mehr im Puffer liegt."""
        return self._handoff(target, at, commit=True)

    def _handoff(self, target: str, at: float | None, commit: bool) -> HandoffResult:
        c = self.cfg

        def no(reason: str, msg: str, **kw) -> HandoffResult:
            if commit:
                self._event("handoff_refused", target=target, reason=reason)
            return HandoffResult(False, reason, msg, target, buffer_s=self._buffer_report(), **kw)

        if target not in self._sources:
            return no("unknown_source", f"Quelle {target!r} ist nicht registriert")
        if not self._segments:
            return no("no_active_source", "Es gibt noch keine aktive Quelle (set_active)")
        s = self._sources[target]
        if self._segments[-1].src == target:
            return no("already_active", f"{target!r} ist schon aktiv oder bereits vorgemerkt")
        act_lag = self.lag_of(self._active) if self._active else 0.0
        # Lag des Ziels relativ zur Zeitachse: Schätzung, sonst Programmquelle + RTT-Hinweis
        if s.tracker.usable:
            lag_s = s.tracker.predict(self._wall) / self.fs
            src = "Schätzung"
        elif s.rtt_hint_s is not None:
            lag_s = act_lag / self.fs + s.rtt_hint_s
            src = "RTT-Hinweis"
        else:
            lag_s = None
            src = ""
        if self._ctl is not None and lag_s is not None:
            out = self._outlook(lag_s)
            if not out.ok:
                b_now = self.buffer_s
                if out.wait_s is None:
                    when = "mit den Reglerwerten nicht erreichbar"
                else:
                    when = f"voraussichtlich frei in {out.wait_s:.0f} s"
                return no("buffer_too_low",
                          f"Puffer {b_now:.2f} s: nach der Übergabe an {target!r} (Lag +"
                          f"{out.lag_growth_s * 1e3:.0f} ms, {src}) blieben {out.buffer_after_s:.2f} s, "
                          f"Untergrenze {c.min_buffer_s:.2f} s; {when}",
                          lag_s=lag_s, buffer_after_s=out.buffer_after_s, wait_s=out.wait_s)
        elif lag_s is not None and lag_s + c.reserve_s > c.buffer_s:
            return no("rtt_exceeds_buffer",
                      f"Lag von {target!r} ({lag_s * 1e3:.0f} ms, {src}) plus Reserve "
                      f"({c.reserve_s * 1e3:.0f} ms) übersteigt den Puffer D = {c.buffer_s * 1e3:.0f} ms",
                      lag_s=lag_s, required_s=lag_s + c.reserve_s)
        if not s.tracker.locked:
            return no("not_locked", f"Für {target!r} gibt es noch keine sichere RTT-Schätzung",
                      lag_s=lag_s)
        if s.tracker.reacquiring:
            return no("not_locked", f"Rückkanal von {target!r} hat gewechselt, die RTT wird neu eingemessen",
                      lag_s=lag_s)
        stale = None if s.tracker.last_update is None else (self._wall - s.tracker.last_update) / self.fs
        if stale is not None and stale > c.max_stale_s and not s.is_reference:
            return no("estimate_stale", f"Letzte gute Messung für {target!r} ist {stale:.1f} s alt",
                      lag_s=lag_s)
        if s.silence_run >= self._sil_timeout:
            return no("target_silent", f"{target!r} liefert gerade Stille", lag_s=lag_s)
        t_next_out = self.output_time
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
        after = self._outlook(lag_s).buffer_after_s if self._ctl is not None else None
        return HandoffResult(True, "ok", "Übergabe vorgemerkt" if commit else "Übergabe möglich",
                             target, t_switch=t_sw, lag_s=lag_s, required_s=lag_s + c.reserve_s,
                             buffer_s=self._buffer_report(), buffer_after_s=after)

    # ------------------------------------------------------------------ Verarbeitung

    def process(self, blocks_by_source: dict[str, np.ndarray], n: int | None = None) -> np.ndarray:
        """Einen Block verarbeiten; gibt P(T) für T = w − D zurück (Stille, solange T < 0)."""
        for name in blocks_by_source:
            if name not in self._sources:
                raise ValueError(f"unbekannte Quelle {name!r}")
        if self._master is not None:
            return self._process_follower(blocks_by_source, n)
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
        if self._stage is None:
            self._t_blk = w0 - self._D
            self._check_fallback(n)
            out = self._compose(self._t_blk, n, count=True).astype(np.float32)
        else:
            self._t_blk = self._stage.pos_int
            self._check_fallback(n)
            out = self._render_elastic(n).astype(np.float32)
        self._prune()
        return out

    def _render_elastic(self, n: int) -> np.ndarray:
        """Ausgabeblock mit geregelter Rate; ohne aktive Quelle bleibt die Rate 1."""
        st, ctl = self._stage, self._ctl
        pos = st.position
        self._blk_start = (st.pos_int, st.pos_frac)
        r0 = 1.0 + ctl.dev
        if self._active is not None:
            ctl.step(self._buffer_at(pos + st.advance_of(n, r0, r0)), n / self.fs)
        r1 = 1.0 + ctl.dev
        out = st.render(lambda t0, m: self._compose(t0, m, count=True), n, r0, r1)
        self.last_block = BlockInfo(n, pos, r0, r1, self._buffer_at(st.position))
        return out

    # ------------------------------------------------------------------ Folger / Umlabeln

    def follow(self, master: "RelayMixer") -> None:
        """Folger-Modus für einen zweiten Kanal (Stereo: Mitte im Master, Seite im Folger).

        Der Folger misst und entscheidet nichts: Lag-Schätzer (dasselbe ``LagTracker``-Objekt),
        Umschaltliste, aktive Quelle, Ausgabeposition und Rate kommen vom Master, der **vor** dem
        Folger mit gleich langen Blöcken verarbeitet. Gleiche Konfiguration und gleiche Quellen
        (Reihenfolge egal) sind Voraussetzung. Die Ausgabe ist dadurch sample-genau auf das
        Raster des Masters ausgerichtet; die Summe aus beiden Kanälen (L = M+S, R = M−S) bleibt
        phasenrichtig."""
        if master is self or master.cfg.sample_rate != self.cfg.sample_rate:
            raise ValueError("ungültiger Master")
        self._master = master

    def _process_follower(self, blocks: dict[str, np.ndarray], n: int | None) -> np.ndarray:
        m = self._master
        if n is None:
            n = len(next(iter(blocks.values()))) if blocks else self.cfg.block_size
        if m._wall != self._wall + n:
            raise RuntimeError("Master muss denselben Block vor dem Folger verarbeiten")
        for name, s in self._sources.items():
            blk = blocks.get(name)
            if blk is None:
                blk = np.zeros(n, dtype=np.float32)
            elif len(blk) != n:
                raise ValueError("alle Blöcke müssen gleich lang sein")
            s.ring.write(np.asarray(blk, dtype=np.float32))
            if name in m._sources:
                s.tracker = m._sources[name].tracker
        self._wall = m._wall
        self._segments, self._active, self._previous = m._segments, m._active, m._previous
        self._cur_seg = m._cur_seg
        if self._stage is None or m.last_block is None:
            self._t_blk = m._t_blk
            return self._compose(self._t_blk, n, count=False).astype(np.float32)
        i, f = m._blk_start
        self._stage.pos_int, self._stage.pos_frac = i, f
        lb = m.last_block
        self._t_blk = i
        return self._stage.render(lambda t0, k: self._compose(t0, k, count=False),
                                  n, lb.r0, lb.r1).astype(np.float32)

    def retime(self, delta: int) -> None:
        """Programmzeit umlabeln: T' = T + ``delta`` (ganze Samples). Rein buchhalterisch: alle Lags
        sinken um ``delta``, Umschaltpunkte, Ausgabeposition und Schätzer-Verlauf wandern mit, die
        Leseposition in den Quell-Ringen (Wanduhr) bleibt exakt gleich - das Audio ändert sich nicht.
        Dient der Verankerung der Zeitachse an der Wanduhr (live/drift.py). Nicht für Folger."""
        if self._master is not None:
            raise RuntimeError("retime am Master aufrufen; der Folger übernimmt den Stand")
        d = int(delta)
        if d == 0:
            return
        for s in self._sources.values():
            s.tracker.shift(d)
        for g in self._segments:
            g.t_start += d
        self._t_blk += d
        if self._stage is not None:
            self._stage.pos_int += d
        if self.last_block is not None:
            self.last_block = BlockInfo(self.last_block.n, self.last_block.pos + d, self.last_block.r0,
                                        self.last_block.r1, self.last_block.buffer_s)
        self._blk_start = (self._blk_start[0] + d, self._blk_start[1])
        self._event("retime", delta=d)

    def read_program(self, t0: int, n: int, source: str | None = None) -> np.ndarray:
        """Programm P(T) für T in [t0, t0+n) mit dem heutigen Stand (Rückkanal/Monitor).

        Mit ``source`` kommt statt des Programms der auf T ausgerichtete Feed genau dieser Quelle.
        Der übernehmende DJ bekommt als Rückkanal den Feed der abgebenden Quelle, solange er
        seinen Übergang mischt; das Programm selbst käme nach der Übergabe aus seinem eigenen
        Feed (Rückkopplung). Was noch nicht angekommen ist, kommt als Null."""
        return self._compose(int(t0), int(n), count=False, only=source)

    def read_feed(self, source: str, w0: int, n: int) -> np.ndarray:
        """Roh-Feed einer Quelle nach Wanduhr-Index ``[w0, w0+n)`` (kein Lag, keine Resampling-
        Korrektur). So kommt der Rückkanal beim DJ an: die abgebende Quelle wird unverändert zurück-
        geschickt; was noch nicht angekommen oder schon überschrieben ist, kommt als Null."""
        return self._get(source).ring.read(int(w0), int(n))

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
        """Programmzeit des nächsten Ausgabe-Samples (ohne Regelung = w − D)."""
        if self._stage is None:
            return self._wall - self._D
        return self._stage.pos_int

    @property
    def output_position(self) -> float:
        """Wie ``output_time``, aber mit Bruchteil (Samples)."""
        if self._stage is None:
            return float(self._wall - self._D)
        return self._stage.position

    @property
    def rate(self) -> float:
        """Aktuelles Ausgabe-Tempo r (1 = Echtzeit; < 1 = langsamer, der Puffer füllt sich)."""
        return 1.0 + self._ctl.dev if self._ctl is not None else 1.0

    @property
    def buffer_s(self) -> float:
        """Puffer B in Sekunden: Live-Rand der aktiven Quelle minus Ausgabeposition."""
        return self._buffer_at(self.output_position)

    def _buffer_at(self, pos: float) -> float:
        lag = self.lag_of(self._active) if self._active else 0.0
        return (self._wall - lag - pos) / self.fs

    def _buffer_report(self) -> float:
        return self.buffer_s if self._ctl is not None else self.cfg.buffer_s

    def _outlook(self, lag_target_s: float) -> HandoffOutlook:
        """Pufferkriterium für eine Übergabe an eine Quelle mit Lag ``lag_target_s`` (absolut):
        Puffer danach = Live-Rand mit neuem Lag minus Ausgabeposition ≥ ``min_buffer_s``."""
        c = self.cfg
        after = self._buffer_at_lag(lag_target_s)
        growth = lag_target_s - (self.lag_of(self._active) / self.fs if self._active else 0.0)
        if after >= c.min_buffer_s:
            return HandoffOutlook(True, after, growth, 0.0)
        # nötiger Puffer jetzt, damit nach dem Zuwachs min_buffer_s bleibt
        need = self.buffer_s + (c.min_buffer_s - after)
        return HandoffOutlook(False, after, growth,
                              self._ctl.time_to(self.buffer_s, need))

    def _buffer_at_lag(self, lag_s: float) -> float:
        return (self._wall - lag_s * self.fs - self.output_position) / self.fs

    def _expected_lag_s(self, s: _Source) -> float | None:
        if s.tracker.usable:
            return s.tracker.predict(self._wall) / self.fs
        if s.rtt_hint_s is not None:
            return (self.lag_of(self._active) if self._active else 0.0) / self.fs + s.rtt_hint_s
        return None

    def _buffer_status(self) -> BufferStatus:
        c = self.cfg
        lag = self.lag_of(self._active) / self.fs if self._active else 0.0
        r = self.rate
        b = self.buffer_s
        delay = (self._wall - self.output_position) / self.fs
        if self._ctl is None:
            return BufferStatus(False, b, None, None, 1.0, c.bpm_ref, delay, lag,
                                release=False, wait_s=None, eta_target_s=None)
        pending = self._segments[-1].src if self._segments else None
        outlooks: dict[str, HandoffOutlook] = {}
        for name, s in self._sources.items():
            if name == self._active or name == pending:
                continue
            lag_t = self._expected_lag_s(s)
            if lag_t is not None:
                outlooks[name] = self._outlook(lag_t)
        waits = [o.wait_s for o in outlooks.values() if o.wait_s is not None]
        wait = min(waits) if waits else None
        if abs(b - c.target_buffer_s) <= (self._ctl.rest_s if self._ctl.engaged else c.deadband_s):
            eta = 0.0
        else:
            lo, hi = self._ctl.reach_limits()
            eta = self._ctl.time_to(b, lo if b < c.target_buffer_s else hi)
        return BufferStatus(True, b, c.target_buffer_s, c.min_buffer_s, r, c.bpm_ref * r, delay,
                            lag, release=any(o.ok for o in outlooks.values()), wait_s=wait,
                            eta_target_s=eta, handoff=outlooks)

    def lag_of(self, name: str, wall: float | None = None) -> float:
        """Lag der Quelle in Samples (Schätzung, mit Drift fortgeschrieben)."""
        return self._get(name).tracker.predict(self._wall if wall is None else wall)

    def lag_params(self, name: str) -> tuple[float, float, float]:
        """(lag in Samples bei t_ref, Drift in Samples je Sample, t_ref als Wanduhr)."""
        return self._lag_params(self._get(name))

    def source_info(self, name: str) -> SourceInfo:
        s = self._get(name)
        tr = s.tracker
        ref = self._active if (self._active and self._active != name) else self._previous
        rtt = None
        if tr.usable and ref is not None and ref != name:
            rtt = (tr.predict(self._wall) - self.lag_of(ref)) / self.fs
        return SourceInfo(
            name=name, locked=tr.usable,
            lag_s=tr.predict(self._wall) / self.fs if tr.usable else None,
            rtt_s=rtt, drift_ppm=tr.rate * 1e6 if tr.usable else None,
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
            sources={k: self.source_info(k) for k in self._sources},
            buffer=self._buffer_status())

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
            if s.name != self._active:
                s.tracker.reset()
                s.is_reference = False
        s.silence_run = n - 1 - int(loud[-1])
        s.had_signal = True
        s.level_db = 20.0 * np.log10(float(np.sqrt(float(blk.dot(blk)) / n)) + 1e-12)

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

    def _compose(self, t0: int, n: int, count: bool, only: str | None = None) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        weights = {self._get(only).name: 1.0} if only else self._weights(t0, n)
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
        """Die Umschaltung gilt, sobald der Live-Rand der bisher aktiven Quelle ihren Punkt passiert.
        Es geht nur vorwärts in der Segmentliste."""
        if not self._segments or self._active is None:
            return
        for _ in range(8):
            cur = next((i for i, g in enumerate(self._segments) if g is self._cur_seg), None)
            if cur is None:
                cur = next((i for i, g in enumerate(self._segments) if g.src == self._active), 0)
                self._cur_seg = self._segments[cur]
            if cur + 1 >= len(self._segments):
                return
            nxt = self._segments[cur + 1]
            t_raw = self._wall - self.lag_of(self._active)
            if nxt.t_start > t_raw:
                return
            old = self._active
            self._sources[old].last_active_wall = self._wall
            self._previous, self._active, self._cur_seg = old, nxt.src, nxt
            self._silent_flag = False
            self._reacquire_waiting()
            self._event("handoff_effective", source=nxt.src, previous=old, t_switch=nxt.t_start)

    def _reacquire_waiting(self) -> None:
        """Jede wartende Quelle hört den Feed der aktiven (Rückkanal); wechselt die aktive Quelle, springt
        ihr Lag um den Lag-Unterschied der beiden aktiven Quellen. Die Nachführung sucht nur ±gate um die
        Vorhersage und fände den neuen Lag nie (S3788) - also alle wartenden Quellen neu einrasten, die
        bisherige Referenzquelle eingeschlossen."""
        start = self._wall + int(self.cfg.reacquire_delay_s * self.fs)
        for s in self._sources.values():
            if s.name == self._active:
                s.tracker.reacquiring = False   # Rückfall auf eine Quelle im Neu-Einrasten: ihr Lag bleibt
                s.est_from = None
                continue
            s.is_reference = False
            s.tracker.begin_reacquire()
            s.est_from = start

    def _still_rendered(self, name: str) -> bool:
        """Liest die Ausgabe noch ein Segment dieser Quelle (Übergang im Puffer)? Dann darf ihr Lag
        nicht springen."""
        t_out = self.output_time
        segs = self._segments
        for j, g in enumerate(segs):
            if g.src == name and (j + 1 >= len(segs) or segs[j + 1].t_start + segs[j + 1].fade > t_out):
                return True
        return False

    def _prune(self) -> None:
        horizon = self._t_blk - int(self.cfg.ring_s * self.fs)
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
            if s.est_from is not None and self._wall - s.est_from < int(c.est_min_window_s * self.fs):
                continue
            if s.tracker.reacquiring and self._still_rendered(s.name):
                continue
            self._estimate(s)

    def _estimate(self, s: _Source) -> None:
        c = self.cfg
        fs = self.fs
        w_end = self._wall
        start = s.signal_start if s.est_from is None else max(s.signal_start, s.est_from)
        n_win = min(int(c.est_window_s * fs), w_end - start)
        if n_win < int(c.est_min_window_s * fs):
            return
        w0 = w_end - n_win
        t_c = w0 + n_win / 2.0
        if s.tracker.locked and abs(s.tracker.rate) > 2e-6:
            # Drift im Fenster herausrechnen: x(w) mit der bisher geschätzten Rate entzerren,
            # sonst verschmiert der Korrelationspeak über rate·Fensterlänge
            pos = w0 + np.arange(n_win) + s.tracker.rate * (np.arange(n_win) - n_win / 2.0)
            x = s.ring.read_frac(pos).astype(np.float32)
        else:
            x = s.ring.read(w0, n_win)
        if float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) < self._sil_thr:
            return
        act_lag = self.lag_of(self._active)
        if s.tracker.locked and not s.tracker.reacquiring:
            pred = s.tracker.predict(t_c)
            lo = int(np.floor(pred - s.tracker.gate))
            hi = int(np.ceil(pred + s.tracker.gate))
        else:
            lo = int(np.ceil(act_lag + c.min_rtt_s * fs))
            hi = int(np.ceil(act_lag + c.search_max_s * fs))
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
            return          # Maximum am Rand: der wahre Peak liegt außerhalb des Suchbereichs
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
        # Der Crossfade soll enden, wo die Quelle verstummt ist: Beginn eine Fade-Länge früher,
        # soweit der Ausgabe-Rand es noch erlaubt (der Puffer D macht das rückwirkend möglich).
        t_sw = max(int(t_sil) - self._fade, self._t_blk, self._segments[-1].t_start)
        self._segments.append(_Segment(t_sw, prev.name, self._fade))
        self._fallback = True
        self._silent_flag = False
        self._event("fallback", source=a, to=prev.name, t_switch=t_sw)
        self._update_active()
