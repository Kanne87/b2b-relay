"""Kennzahlen aus dem Mixer-Zustand für die Status-API."""

from __future__ import annotations

import numpy as np

from ..rtt import LagTracker


def lag_jitter_ms(tracker: LagTracker, fs: int) -> float | None:
    """Streuung der Lag-Messungen um die angepasste Gerade (Standardabweichung, ms) über die
    letzten ~20 s. Das ist der Messjitter der RTT-Schätzung (Netzjitter plus Schätzrauschen).
    ``None``, solange weniger als vier angenommene Messungen vorliegen."""
    hist = list(tracker._hist)      # (Fenstermitte, Verzögerung) - nur lesen, Worker-Thread
    if len(hist) < 4:
        return None
    t = np.array([h[0] for h in hist], dtype=np.float64)
    d = np.array([h[1] for h in hist], dtype=np.float64)
    if t.max() - t.min() < 1.0:
        return float(d.std() / fs * 1e3)
    tm = t.mean()
    slope, icpt = np.polyfit(t - tm, d, 1)
    return float((d - (icpt + slope * (t - tm))).std() / fs * 1e3)
