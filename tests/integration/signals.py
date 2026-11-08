"""Testsignale und Beat-Erkennung für den Integrationstest (nur numpy).

* Jeder DJ spielt 4-on-the-floor-Kicks bei BPM 174 mit eigener Klangfarbe (Sweep-Bereich, Zerfall,
  Oberton) und ein leises Pad aus zwei Sinustönen (damit das Programm nie digital still ist - Lücken
  im Mitschnitt sind dann echte Lücken).
* Alle Kicks haben denselben Anfang (Kosinus, 0,5 ms linearer Anstieg): der Onset ist über
  Klangfarben hinweg vergleichbar und wird als erster Schwellendurchgang des Betrags bestimmt
  (lineare Interpolation, ~20 µs genau).
* ``PhaseLock`` bestimmt aus dem gehörten Feed das Raster (Kreismittel der Kick-Zeiten modulo Beat);
  der Übernehmende spielt seine Kicks darauf - wie der DJ, der nach Gehör einstartet (sim/).
"""

from __future__ import annotations

import numpy as np

FS = 48000
BPM = 174.0
BEAT = FS * 60.0 / BPM                      # Frames je Beat (16551,72…)
KICK_PEAK = 0.4
PAD_LEVEL = 0.03
ONSET_THR = 0.15                            # Schwelle auf |x| (Kick-Peak 0,4; Pad ≤ 0,06)

TIMBRES = {
    #        f_start  f_end  decay_s  harmonic
    "A": dict(f0=150.0, f1=48.0, tau=0.075, h2=0.15, pad=(220.0, 330.0)),
    "B": dict(f0=125.0, f1=55.0, tau=0.100, h2=0.25, pad=(247.0, 370.0)),
    "C": dict(f0=185.0, f1=42.0, tau=0.060, h2=0.10, pad=(262.0, 392.0)),
}


def kick(name: str, fs: int = FS) -> np.ndarray:
    t = TIMBRES.get(name, TIMBRES["A"])
    n = int(0.45 * fs)
    x = np.arange(n) / fs
    f = t["f1"] + (t["f0"] - t["f1"]) * np.exp(-x / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / fs
    env = np.minimum(x / 0.0005, 1.0) * np.exp(-x / t["tau"])
    y = env * (np.cos(ph) + t["h2"] * np.cos(2 * ph))
    y *= KICK_PEAK / np.abs(y).max()
    return y.astype(np.float32)


def pad(name: str, start: int, n: int, fs: int = FS) -> np.ndarray:
    f1, f2 = TIMBRES.get(name, TIMBRES["A"])["pad"]
    i = (np.arange(start, start + n) / fs).astype(np.float64)
    return (PAD_LEVEL * (np.sin(2 * np.pi * f1 * i) + np.sin(2 * np.pi * f2 * i)) * 0.5).astype(np.float32)


def render_track(name: str, seconds: float, phase_frames: float = 0.0, fs: int = FS) -> np.ndarray:
    """Mono-Track (Kicks auf dem Raster ``phase_frames + k·BEAT`` plus Pad) - für Tests und Referenz."""
    n = int(seconds * fs)
    out = pad(name, 0, n, fs)
    k = kick(name, fs)
    t = phase_frames
    while t < n:
        i = int(round(t))
        m = min(len(k), n - i)
        if m > 0 and i >= 0:
            out[i:i + m] += k[:m]
        t += BEAT
    return out


def detect_onsets(x: np.ndarray, thr: float = ONSET_THR, refractory: float = 0.08, fs: int = FS) -> np.ndarray:
    """Kick-Onsets (Frames, mit Bruchteil) als erster Durchgang von |x| über ``thr`` nach einer
    Ruhephase von ``refractory`` Sekunden unterhalb der Schwelle."""
    a = np.abs(x.astype(np.float64))
    above = a >= thr
    if not above.any():
        return np.empty(0)
    idx = np.flatnonzero(above & ~np.concatenate(([False], above[:-1])))
    out = []
    last_hi = -10 ** 12
    hi_idx = np.flatnonzero(above)
    # letzter Index über der Schwelle vor jedem Kandidaten
    pos = np.searchsorted(hi_idx, idx) - 1
    for i, p in zip(idx, pos):
        last_hi = hi_idx[p] if p >= 0 else -10 ** 12
        if i - last_hi < refractory * fs and out:
            continue
        # lineare Interpolation des Schwellendurchgangs
        a0 = a[i - 1] if i > 0 else 0.0
        frac = (thr - a0) / (a[i] - a0) if a[i] > a0 else 0.0
        out.append(i - 1 + frac if i > 0 else float(i))
    return np.array(out)


def grid_phase(onsets: np.ndarray, beat: float = BEAT) -> tuple[float, float]:
    """Kreismittel von onsets mod beat → (Phase in Frames [0, beat), Streuung R in 0..1; 1 = perfekt)."""
    if len(onsets) == 0:
        return 0.0, 0.0
    ang = 2 * np.pi * (np.asarray(onsets) % beat) / beat
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    return float((np.arctan2(s, c) % (2 * np.pi)) * beat / (2 * np.pi)), float(np.hypot(c, s))


def wrap(d: float, beat: float = BEAT) -> float:
    """Differenz auf (−beat/2, beat/2] falten."""
    return (d + beat / 2) % beat - beat / 2


class PhaseLock:
    """Hält die Rasterphase des gehörten Feeds (Frames, globaler Zähler)."""

    def __init__(self, min_onsets: int = 6, min_r: float = 0.9) -> None:
        self.min_onsets, self.min_r = min_onsets, min_r
        self.phase: float | None = None

    def update(self, onsets: np.ndarray) -> bool:
        if len(onsets) < self.min_onsets:
            return False
        ph, r = grid_phase(onsets[-12:])
        if r < self.min_r:
            return False
        if self.phase is None:
            self.phase = ph
        else:
            self.phase = (self.phase + 0.3 * wrap(ph - self.phase)) % BEAT
        return True
