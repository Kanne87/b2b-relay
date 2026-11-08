"""Synthetische Drum-and-Bass-Tracks (174 BPM) für die Simulation.

Ein Track besteht aus zwei Stems, damit das DJ-Modell einen Bass-Swap fahren kann, ohne zu
filtern: ``low`` (Kick + Bass-Sinus) und ``high`` (Snare, Hi-Hat, Stabs, Dither). Muster,
Tonhöhen und Klangfarben hängen vom Seed ab. Die Bar-Muster variieren von Takt zu Takt, damit
die Kreuzkorrelation eindeutig bleibt (ein reiner Vier-Takt-Loop wäre mehrdeutig).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FS = 48000
BPM = 174.0


def beat_samples(fs: int = FS, bpm: float = BPM) -> float:
    return fs * 60.0 / bpm


def bar_samples(fs: int = FS, bpm: float = BPM) -> float:
    return 4.0 * beat_samples(fs, bpm)


@dataclass
class Track:
    """Gerenderter Track; Sample 0 ist der Anfang von Takt 1."""

    seed: int
    bars: int
    fs: int = FS
    bpm: float = BPM
    low: np.ndarray = field(init=False, repr=False)
    high: np.ndarray = field(init=False, repr=False)
    kick_times: np.ndarray = field(init=False, repr=False)   # Samples (Bruchteil) seit Trackanfang

    def __post_init__(self) -> None:
        self._render()

    @property
    def n(self) -> int:
        return self.low.shape[0]

    @property
    def mix(self) -> np.ndarray:
        return self.low + self.high

    def _render(self) -> None:
        fs = self.fs
        rng = np.random.default_rng([self.seed, 174])
        step = bar_samples(fs, self.bpm) / 16.0
        tail = int(0.6 * fs)
        n = int(np.ceil(self.bars * 16 * step)) + tail
        low = np.zeros(n, dtype=np.float64)
        high = np.zeros(n, dtype=np.float64)
        kicks: list[float] = []

        # Klangfarbe dieses Tracks
        k_f0 = rng.uniform(44.0, 56.0)
        k_sweep = rng.uniform(100.0, 140.0)
        k_tau = rng.uniform(0.014, 0.022)
        k_decay = rng.uniform(0.09, 0.14)
        s_tone = rng.uniform(170.0, 230.0)
        hat_noise = np.diff(rng.standard_normal(int(0.4 * fs) + 1))   # Hochpass-Rauschen
        hat_noise /= np.abs(hat_noise).max()
        scale = [0, 3, 5, 7, 10]
        root = rng.uniform(38.0, 46.0)           # ≈ D1..F#1

        def add(buf: np.ndarray, start: float, sig: np.ndarray) -> None:
            i0 = int(np.ceil(start))
            if i0 >= n:
                return
            m = min(len(sig), n - i0)
            buf[i0:i0 + m] += sig[:m]

        def kick(t_start: float, vel: float) -> None:
            i0 = int(np.ceil(t_start))
            m = min(int(0.5 * fs), n - i0)
            if m <= 0:
                return
            t = (np.arange(i0, i0 + m) - t_start) / fs
            ph = 2 * np.pi * (k_f0 * t + k_sweep * k_tau * (1.0 - np.exp(-t / k_tau)))
            sig = np.exp(-t / k_decay) * np.sin(ph)
            click = np.exp(-t / 0.0012) * np.sin(2 * np.pi * 1800.0 * t)
            low[i0:i0 + m] += vel * (0.8 * sig + 0.18 * click)
            kicks.append(float(t_start))

        def snare(t_start: float, vel: float) -> None:
            i0 = int(np.ceil(t_start))
            m = min(int(0.25 * fs), n - i0)
            if m <= 0:
                return
            t = (np.arange(i0, i0 + m) - t_start) / fs
            noise = rng.standard_normal(m)
            high[i0:i0 + m] += vel * 0.45 * (np.exp(-t / 0.05) * noise * 0.7
                                              + np.exp(-t / 0.06) * np.sin(2 * np.pi * s_tone * t) * 0.6)

        def hat(t_start: float, vel: float, length: float) -> None:
            i0 = int(np.ceil(t_start))
            m = min(int(length * fs * 4), len(hat_noise), n - i0)
            if m <= 0:
                return
            t = np.arange(m) / fs
            off = int(rng.integers(0, len(hat_noise) - m + 1))
            high[i0:i0 + m] += vel * 0.16 * np.exp(-t / length) * hat_noise[off:off + m]

        def stab(t_start: float, freqs: list[float], vel: float) -> None:
            i0 = int(np.ceil(t_start))
            m = min(int(0.3 * fs), n - i0)
            if m <= 0:
                return
            t = (np.arange(i0, i0 + m) - t_start) / fs
            sig = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
            high[i0:i0 + m] += vel * 0.14 * np.exp(-t / 0.07) * (1 - np.exp(-t / 0.002)) * sig

        # Bass: ein Sinus je Takt (ggf. zweite Note ab Schritt 10), nahtlos über Gain-Rampen
        bass_notes: list[tuple[float, float, float]] = []   # (start, end, freq)
        for b in range(self.bars):
            b0 = b * 16 * step
            # Kick: Zwei-Schritt-Grundmuster plus Variation
            kick(b0, 0.95)
            kick(b0 + 10 * step, 0.9)
            if rng.random() < 0.28:
                kick(b0 + 7 * step, 0.55)
            if rng.random() < 0.2:
                kick(b0 + 14 * step, 0.5)
            # Snare
            snare(b0 + 4 * step, 0.95)
            snare(b0 + 12 * step, 0.95)
            for s in (2, 7, 9, 15):
                if rng.random() < 0.12:
                    snare(b0 + s * step, 0.3)
            if b % 8 == 7:
                for s in (13, 14, 15):
                    snare(b0 + s * step, 0.5)
            # Hi-Hats
            for s in range(0, 16, 2):
                if rng.random() > 0.15:
                    hat(b0 + s * step, rng.uniform(0.5, 1.0), 0.03)
            for s in (3, 6, 11, 15):
                if rng.random() < 0.3:
                    hat(b0 + s * step, rng.uniform(0.4, 0.9), 0.012)
            if rng.random() < 0.35:
                hat(b0 + 6 * step, 0.8, 0.09)
            # Stabs
            for s in range(16):
                if rng.random() < 0.09:
                    fr = [float(rng.choice([220.0, 261.6, 329.6, 392.0, 440.0, 523.3, 659.3, 784.0]))
                          for _ in range(int(rng.integers(2, 4)))]
                    stab(b0 + s * step, fr, rng.uniform(0.5, 1.0))
            f1 = root * 2 ** (scale[int(rng.integers(0, len(scale)))] / 12.0)
            bass_notes.append((b0, b0 + 16 * step, f1))
            if rng.random() < 0.5:
                f2 = root * 2 ** (scale[int(rng.integers(0, len(scale)))] / 12.0)
                bass_notes[-1] = (b0, b0 + 10 * step, f1)
                bass_notes.append((b0 + 10 * step, b0 + 16 * step, f2))
        # Bass-Noten zusammensetzen (Phase pro Note fortlaufend, kurze Rampen)
        ramp = int(0.004 * fs)
        for (a, z, f) in bass_notes:
            i0, i1 = int(round(a)), min(int(round(z)) + ramp, n)
            t = np.arange(i0, i1) / fs
            g = np.ones(i1 - i0)
            g[:ramp] = np.linspace(0, 1, ramp)
            g[-ramp:] = np.linspace(1, 0, ramp)
            low[i0:i1] += 0.3 * g * (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t))
        # Dither: nie digitale Stille
        high += rng.standard_normal(n) * 3e-4
        scale_out = 0.55
        self.low = (low * scale_out).astype(np.float32)
        self.high = (high * scale_out).astype(np.float32)
        self.kick_times = np.array(sorted(kicks), dtype=np.float64)
