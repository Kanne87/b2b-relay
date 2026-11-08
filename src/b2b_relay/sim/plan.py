"""Drehbuch der Simulation in Programmzeit (Samples auf der Zeitachse T).

Szenario A→B→A…: Jeder DJ mixt nach seiner Übernahme zwei Tracks selbst hinein, dann hört der
andere DJ das Programm (arm), startet seinen Track phasengenau dazu (Übergabe-Zeitpunkt) und
mischt über N Takte. Menschlicher Fehler: jeder neue Track startet um ``δ`` neben dem Takt des
gehörten Programms; der Taktraster wandert dadurch wie ein Random Walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .synth import Track, bar_samples

SWAP_S = 0.05        # Dauer des Bass-Swaps
MUTE_AFTER_S = 2.0   # nach der Übergabe spielt der abgebende DJ noch kurz weiter
MUTE_RAMP_S = 0.2


@dataclass
class PlanConfig:
    handoffs: int = 10
    names: tuple[str, str] = ("A", "B")
    mix_bars: int = 8
    hold_bars: int = 2
    arm_bars: int = 6
    human_error_ms: float = 5.0
    seed: int = 1
    fs: int = 48000


@dataclass
class PTrack:
    idx: int
    dj: str
    start: float                  # Programmzeit des Track-Anfangs (inkl. menschlichem Fehler)
    kind: str                     # initial | solo | handoff
    human_s: float = 0.0
    mix_bars: int = 0             # Länge der Einblendung
    nominal_start: float = 0.0    # Taktlinie ohne menschlichen Fehler
    track: Track | None = field(default=None, repr=False)
    nxt: "PTrack | None" = field(default=None, repr=False)
    mute_at: float | None = None  # Programmzeit, ab der der DJ ausblendet (nach Übergabe)

    @property
    def swap_at(self) -> float:
        return self.start  # wird über Plan.bar gesetzt, s. swap_point()


@dataclass
class PHandoff:
    index: int
    src: str
    dst: str
    t_arm: float                  # Programmzeit, ab der dst das Programm hört
    t_h: float                    # Übergabe-Zeitpunkt (= Beginn von dsts Einblendung)
    track: PTrack = field(repr=False, default=None)    # dsts neuer Track
    old: PTrack = field(repr=False, default=None)      # das Programm, das dst hört


class Plan:
    def __init__(self, cfg: PlanConfig) -> None:
        self.cfg = cfg
        self.fs = cfg.fs
        self.bar = bar_samples(cfg.fs)
        self.tracks: list[PTrack] = []
        self.handoffs: list[PHandoff] = []
        self._build()

    def swap_point(self, t: PTrack) -> float:
        return t.start + 0.5 * t.mix_bars * self.bar

    def _build(self) -> None:
        cfg, bar = self.cfg, self.bar
        rng = np.random.default_rng([cfg.seed, 7])
        N, H = cfg.mix_bars, cfg.hold_bars

        def delta() -> float:
            return float(rng.uniform(-1.0, 1.0)) * cfg.human_error_ms * 1e-3

        a, b = cfg.names
        cur = PTrack(0, a, 0.0, "initial")
        self.tracks.append(cur)
        active = a
        for h in range(cfg.handoffs):
            other = b if active == a else a
            for _ in range(2):      # zwei eigene Übergänge
                nominal = cur.start + (max(cur.mix_bars, 0) + H) * bar
                d = delta()
                t = PTrack(len(self.tracks), active, nominal + d * self.fs, "solo", d, N, nominal)
                cur.nxt = t
                self.tracks.append(t)
                cur = t
            t_arm = cur.start + (N + 1) * bar
            t_h = cur.start + (N + 1 + cfg.arm_bars) * bar
            d = delta()
            new = PTrack(len(self.tracks), other, t_h + d * self.fs, "handoff", d, N, t_h)
            cur.mute_at = t_h + MUTE_AFTER_S * self.fs
            self.handoffs.append(PHandoff(h, active, other, t_arm, t_h, new, cur))
            cur.nxt = new
            self.tracks.append(new)
            cur = new
            active = other
        self.t_end = cur.start + (N + 4) * bar + 4 * self.fs
        bars_needed = 3 * N + 2 * H + cfg.arm_bars + 8
        for i, t in enumerate(self.tracks):
            t.track = Track(seed=cfg.seed * 1000 + i, bars=bars_needed, fs=self.fs)

    def own_gains(self, t: PTrack, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(low, high)-Gains eines Tracks im Master seines DJs (ohne Rückkanal-Anteil)."""
        fs, bar = self.fs, self.bar
        swap_len = SWAP_S * fs

        def smooth(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)

        if t.kind == "initial":
            gl = np.ones_like(T)
            gh = np.ones_like(T)
        else:
            span = t.mix_bars * bar
            gh = smooth((T - t.start) / span)
            gl = smooth((T - self.swap_point(t)) / swap_len)
        nx = t.nxt
        if nx is not None:
            if nx.kind == "solo":
                span = nx.mix_bars * bar
                gh = gh * (1.0 - smooth((T - nx.start) / span))
                gl = gl * (1.0 - smooth((T - self.swap_point(nx)) / swap_len))
            else:   # Übergabe an den anderen DJ: nach kurzer Weile ausblenden
                g = 1.0 - smooth((T - t.mute_at) / (MUTE_RAMP_S * fs))
                gh = gh * g
                gl = gl * g
        return gl, gh

    def echo_gains(self, ho: PHandoff, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(low, high)-Gains des Rückkanals im Master des übernehmenden DJs."""
        fs, bar = self.fs, self.bar
        t = ho.track
        swap_len = SWAP_S * fs

        def smooth(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)

        gh = 1.0 - smooth((T - t.start) / (t.mix_bars * bar))
        gl = 1.0 - smooth((T - self.swap_point(t)) / swap_len)
        return gl, gh

    def echo_window(self, ho: PHandoff) -> tuple[int, int]:
        t = ho.track
        return int(np.floor(ho.t_arm)), int(np.ceil(t.start + t.mix_bars * self.bar)) + 1
