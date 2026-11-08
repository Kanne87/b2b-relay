"""RTT-/Versatz-Schätzung einer Quelle gegen das Programm per GCC-PHAT plus Nachführung.

``gcc_phat`` misst für ein Fenster des eingehenden Feeds ``x`` den Versatz ``d`` mit
``x[n] ≈ g · P(n − d)``. ``LagTracker`` glättet die Messungen mit einem α-β-Filter, hält die
letzte gute Schätzung samt Driftrate, wenn keine verlässliche Messung kommt, und weist
Ausreißer ab.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DelayMeasurement:
    delay: float        # Samples, Bruchteil per Parabel-Interpolation
    confidence: float   # 0..1
    psr: float          # Peak-to-Sidelobe-Ratio (Standardabweichungen über dem Rauschboden)
    ratio: float        # Peak / zweithöchster Peak außerhalb der Umgebung


def _next_pow2(n: int) -> int:
    return 1 << max(4, (int(n) - 1).bit_length())


def gcc_phat(x: np.ndarray, p: np.ndarray, max_shift: int, *, beta: float = 0.85,
             exclude: int = 48) -> DelayMeasurement | None:
    """Versatz von ``x`` gegen ``p`` per generalisierter Kreuzkorrelation mit PHAT-Gewichtung.

    ``p`` hat die Länge ``len(x) + max_shift``; ``p[k]`` ist das Programm bei Index
    ``x0 − lo − max_shift + k`` (``x0`` = Index von ``x[0]``, ``lo`` = untere Suchgrenze, der
    Aufrufer legt sie fest). Gemessen wird ``delay`` ∈ [0, max_shift] mit
    ``x[n] ≈ P(x0 + n − lo − delay)``. Liefert None bei Stille.
    """
    nx, npg = len(x), len(p)
    if nx < 256 or npg < nx:
        return None
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if not (np.any(x) and np.any(p)):
        return None
    nfft = _next_pow2(npg + 8)
    X = np.fft.rfft(x - x.mean(), nfft)
    P = np.fft.rfft(p - p.mean(), nfft)
    cross = np.conj(X) * P
    mag = np.abs(cross)
    floor = 1e-6 * float(mag.mean()) + 1e-30
    # PHAT-β: β=1 ist reines PHAT, kleinere Werte gewichten energiereiche Bins stärker.
    r = cross / (mag + floor) ** beta * (mag.mean() + floor) ** (beta - 1.0)
    c = np.fft.irfft(r, nfft)
    # c[s] = Σ x[n]·p[n+s], gültig für s in [0, max_shift] → d = max_shift − s
    seg = c[:max_shift + 1]
    k = int(np.argmax(seg))
    peak = float(seg[k])
    if peak <= 0.0:
        return None
    if 0 < k < max_shift:
        a, b, g = seg[k - 1], seg[k], seg[k + 1]
        den = a - 2.0 * b + g
        off = 0.5 * (a - g) / den if den != 0 else 0.0
        off = float(np.clip(off, -1.0, 1.0))
    else:
        off = 0.0
    # Rauschboden aus allen Lags außerhalb der Umgebung des Peaks (voller Kreis)
    mask = np.ones(nfft, dtype=bool)
    lo, hi = k - exclude, k + exclude + 1
    mask[max(lo, 0):max(hi, 0)] = False
    if lo < 0:
        mask[lo:] = False
    rest = c[mask]
    sd = float(rest.std()) + 1e-30
    psr = (peak - float(rest.mean())) / sd
    # zweiter Peak innerhalb des Suchbereichs
    seg2 = seg.copy()
    seg2[max(lo, 0):min(hi, len(seg2))] = -np.inf
    second = float(seg2.max()) if np.isfinite(seg2).any() else 0.0
    ratio = peak / second if second > 0 else 10.0
    conf = float(np.clip((psr - 6.0) / 10.0, 0.0, 1.0)) * float(np.clip((ratio - 1.0) / 0.6, 0.0, 1.0))
    return DelayMeasurement(delay=max_shift - (k + off), confidence=conf, psr=psr, ratio=ratio)


class LagTracker:
    """α-β-Nachführung des Versatzes (Samples) mit Driftrate (Samples je Sample)."""

    def __init__(self, *, alpha: float = 0.45, beta: float = 0.12, gate: float = 144.0,
                 rate_limit: float = 500e-6, min_conf: float = 0.5,
                 reacquire_conf: float = 0.8) -> None:
        self.alpha, self.beta, self.gate = alpha, beta, gate
        self.rate_limit, self.min_conf, self.reacquire_conf = rate_limit, min_conf, reacquire_conf
        self.reset()

    def reset(self) -> None:
        self.lag = 0.0
        self.rate = 0.0
        self.t_ref = 0.0
        self.locked = False
        self.confidence = 0.0
        self.last_update = None      # Wanduhr der letzten angenommenen Messung
        self._cand: tuple[float, float] | None = None
        self._outliers: list[tuple[float, float]] = []
        self.n_updates = 0

    def set_reference(self, lag: float, t: float = 0.0) -> None:
        """Lag per Definition setzen (erste Quelle, Programm-Referenz)."""
        self.lag, self.rate, self.t_ref = float(lag), 0.0, float(t)
        self.locked = True
        self.confidence = 1.0

    def predict(self, t: float) -> float:
        return self.lag + self.rate * (t - self.t_ref)

    def lag_and_rate(self) -> tuple[float, float, float]:
        return self.lag, self.rate, self.t_ref

    def update(self, t_c: float, delay: float, conf: float) -> bool:
        """Messung (Zeitpunkt = Fenstermitte) einarbeiten. True, wenn angenommen."""
        if conf < self.min_conf:
            return False
        if not self.locked:
            if self._cand is not None and abs(delay - self._cand[1]) <= self.gate \
                    and t_c > self._cand[0]:
                t0, d0 = self._cand
                self.rate = float(np.clip((delay - d0) / (t_c - t0), -self.rate_limit, self.rate_limit))
                self.lag, self.t_ref, self.locked = float(delay), float(t_c), True
                self.confidence, self.last_update = conf, t_c
                self.n_updates += 1
                self._cand = None
                return True
            self._cand = (t_c, delay)
            return False
        resid = delay - self.predict(t_c)
        if abs(resid) <= self.gate:
            dt = t_c - self.t_ref
            self.lag = self.predict(t_c) + self.alpha * resid
            if dt > 0:
                self.rate = float(np.clip(self.rate + self.beta * resid / dt,
                                          -self.rate_limit, self.rate_limit))
            self.t_ref = float(t_c)
            self.confidence, self.last_update = conf, t_c
            self._outliers.clear()
            self.n_updates += 1
            return True
        # Ausreißer: nur nach drei untereinander stimmigen, sehr sicheren Messungen neu aufsetzen
        if conf >= self.reacquire_conf:
            if self._outliers and abs(delay - self._outliers[-1][1]) > self.gate:
                self._outliers.clear()
            self._outliers.append((t_c, delay))
            if len(self._outliers) >= 3:
                (t0, d0), (t1, d1) = self._outliers[0], self._outliers[-1]
                self.lag, self.t_ref = float(d1), float(t1)
                self.rate = float(np.clip((d1 - d0) / (t1 - t0), -self.rate_limit, self.rate_limit))
                self.confidence, self.last_update = conf, t_c
                self._outliers.clear()
                self.n_updates += 1
                return True
        return False
