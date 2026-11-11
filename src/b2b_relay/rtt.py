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


def gcc_curve(x: np.ndarray, p: np.ndarray, max_shift: int, *, beta: float = 1.0) -> np.ndarray | None:
    """Kreuzkorrelation mit PHAT-Gewichtung als voller Kreis (Länge ``nfft``); ``c[s] = Σ x[n]·p[n+s]``,
    gültig für ``s`` in [0, max_shift]. Aufteilung wie in ``gcc_phat``. None bei Stille."""
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
    return np.fft.irfft(r, nfft)


def measure_curve(c: np.ndarray, max_shift: int, k_lo: int = 0, k_hi: int | None = None, *,
                  exclude: int = 48) -> DelayMeasurement | None:
    """Messung aus einer Kurve von ``gcc_curve``: Maximum für ``s`` in [k_lo, k_hi] (Teil des Suchbereichs),
    Konfidenz aus Peak-to-Sidelobe (voller Kreis) und Abstand zum zweithöchsten Peak im selben Teilbereich.
    ``delay = max_shift − s`` wie bei ``gcc_phat``."""
    k_hi = max_shift if k_hi is None else k_hi
    k_lo, k_hi = max(0, int(k_lo)), min(max_shift, int(k_hi))
    if k_hi - k_lo < 2:
        return None
    nfft = len(c)
    seg = c[k_lo:k_hi + 1]
    j = int(np.argmax(seg))
    k = k_lo + j
    peak = float(seg[j])
    if peak <= 0.0:
        return None
    if 0 < j < len(seg) - 1:
        a, b, g = seg[j - 1], seg[j], seg[j + 1]
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
    # zweiter Peak innerhalb des (Teil-)Suchbereichs
    seg2 = seg.copy()
    seg2[max(lo - k_lo, 0):max(min(hi - k_lo, len(seg2)), 0)] = -np.inf
    second = float(seg2.max()) if np.isfinite(seg2).any() else 0.0
    ratio = peak / second if second > 0 else 10.0
    conf = float(np.clip((psr - 6.0) / 10.0, 0.0, 1.0)) * float(np.clip((ratio - 1.0) / 0.6, 0.0, 1.0))
    return DelayMeasurement(delay=max_shift - (k + off), confidence=conf, psr=psr, ratio=ratio)


def gcc_phat(x: np.ndarray, p: np.ndarray, max_shift: int, *, beta: float = 1.0,
             exclude: int = 48) -> DelayMeasurement | None:
    """Versatz von ``x`` gegen ``p`` per generalisierter Kreuzkorrelation mit PHAT-Gewichtung.

    ``p`` hat die Länge ``len(x) + max_shift``; ``p[k]`` ist das Programm bei Index
    ``x0 − lo − max_shift + k`` (``x0`` = Index von ``x[0]``, ``lo`` = untere Suchgrenze, der
    Aufrufer legt sie fest). Gemessen wird ``delay`` ∈ [0, max_shift] mit
    ``x[n] ≈ P(x0 + n − lo − delay)``. Liefert None bei Stille.
    """
    c = gcc_curve(x, p, max_shift, beta=beta)
    if c is None:
        return None
    return measure_curve(c, max_shift, exclude=exclude)


class LagTracker:
    """α-β-Nachführung des Versatzes (Samples) mit Driftrate (Samples je Sample)."""

    def __init__(self, *, alpha: float = 0.45, beta: float = 0.12, gate: float = 144.0,
                 rate_limit: float = 500e-6, min_conf: float = 0.5,
                 reacquire_conf: float = 0.8, fit_span: float = 20.0 * 48000,
                 fit_min_span: float = 2.0 * 48000, confirm_span: float = 0.0,
                 relock_span: float = 0.0, accept: float | None = None) -> None:
        self.fit_span, self.fit_min_span = fit_span, fit_min_span
        # Einrasten erst, wenn die bestätigende Messung so weit (Fenstermitten, Samples) nach der ersten liegt:
        # mit mindestens einer Fensterlänge stammen beide aus getrennten Fenstern, ein einzelnes Ereignis im
        # Signal (Knacks, Aussetzer, Signalbeginn) kann nicht beide tragen
        self.confirm_span = confirm_span
        self.relock_span = relock_span      # ebenso für das Neu-Aufsetzen nach stimmigen Ausreißern
        self.alpha, self.beta, self.gate = alpha, beta, gate
        # Annahmebereich um die Vorhersage (Vorgabe: gate). Mit ``accept`` zappelt der Lag in kleinen Stufen
        # (JackTrip-Jitterpuffer: ±1-3 Slots für Bruchteile einer Sekunde): Messungen bis ±2·accept gehen in den
        # Verlauf, die Schätzung ist der Mittelwert der Messungen bis ±accept um ihren Median über ``fit_span``
        # - der mittlere Versatz, den die Musik im Programm hat. Um den Median, nicht um die eigene Schätzung:
        # sonst hielte eine Einrastung in einem hohen Zustand die tiefen dauerhaft draußen (+6…+9 ms, S3904).
        self.tol = gate if accept is None else max(gate, accept)
        self.robust = accept is not None
        self.wide = 2.0 * self.tol if self.robust else self.tol
        self.rate_limit, self.min_conf, self.reacquire_conf = rate_limit, min_conf, reacquire_conf
        self.reset()

    def reset(self) -> None:
        self.lag = 0.0
        self.rate = 0.0
        self.t_ref = 0.0
        self.locked = False
        self.reacquiring = False     # Lag hat sich sprunghaft geändert: neu einrasten, alte Schätzung gilt weiter
        self.confidence = 0.0
        self.last_update = None      # Wanduhr der letzten angenommenen Messung
        self._cand: tuple[float, float] | None = None
        self._outliers: list[tuple[float, float]] = []
        self._hist: list[tuple[float, float]] = []   # angenommene Messungen (t_c, delay)
        self.n_updates = 0

    def set_reference(self, lag: float, t: float = 0.0) -> None:
        """Lag per Definition setzen (erste Quelle, Programm-Referenz)."""
        self.lag, self.rate, self.t_ref = float(lag), 0.0, float(t)
        self._hist.clear()
        self.locked = True
        self.reacquiring = False
        self.confidence = 1.0

    def begin_reacquire(self) -> None:
        """Der Lag springt (z. B. weil die Quelle jetzt einen anderen Rückkanal hört): neu einrasten wie
        beim ersten Mal (zwei stimmige Messungen), ohne die bisherige Schätzung vorher zu verwerfen -
        sie bleibt für bereits geplantes Audio gültig, bis die neue steht. Die Driftrate bleibt (die Uhr
        der Quelle ändert sich nicht)."""
        if self.locked:
            self.reacquiring = True
            self._cand = None
            self._outliers.clear()

    @property
    def usable(self) -> bool:
        """Eingerastet und nicht im Neu-Einrasten: die Schätzung gilt für neues Audio."""
        return self.locked and not self.reacquiring

    def predict(self, t: float) -> float:
        return self.lag + self.rate * (t - self.t_ref)

    def shift(self, d: float) -> None:
        """Programmzeit umlabeln (T' = T + d): jeder Lag sinkt um ``d``, auch die gemerkten Messungen."""
        self.lag -= d
        self._hist = [(t, x - d) for t, x in self._hist]
        self._outliers = [(t, x - d) for t, x in self._outliers]
        if self._cand is not None:
            self._cand = (self._cand[0], self._cand[1] - d)

    def lag_and_rate(self) -> tuple[float, float, float]:
        return self.lag, self.rate, self.t_ref

    def update(self, t_c: float, delay: float, conf: float) -> bool:
        """Messung (Zeitpunkt = Fenstermitte) einarbeiten. True, wenn angenommen."""
        if conf < self.min_conf:
            return False
        if self.reacquiring:
            if self._cand is not None and abs(delay - self._cand[1]) <= self.tol and t_c > self._cand[0]:
                if t_c - self._cand[0] < self.confirm_span:
                    return False            # stimmig, aber noch dasselbe Fenster: Kandidat behalten
                t0, d0 = self._cand
                self.lag, self.t_ref, self.reacquiring = float(delay), float(t_c), False
                self.confidence, self.last_update = conf, t_c
                self.n_updates += 1
                self._cand = None
                self._hist = [(t0, d0), (t_c, delay)]
                return True
            self._cand = (t_c, delay)
            return False
        if not self.locked:
            if self._cand is not None and abs(delay - self._cand[1]) <= self.tol \
                    and t_c > self._cand[0]:
                if t_c - self._cand[0] < self.confirm_span:
                    return False
                t0, d0 = self._cand
                self.rate = float(np.clip((delay - d0) / (t_c - t0), -self.rate_limit, self.rate_limit))
                self.lag, self.t_ref, self.locked = float(delay), float(t_c), True
                self.confidence, self.last_update = conf, t_c
                self.n_updates += 1
                self._cand = None
                self._hist = [(t0, d0), (t_c, delay)]
                return True
            self._cand = (t_c, delay)
            return False
        resid = delay - self.predict(t_c)
        if abs(resid) <= self.wide:
            dt = t_c - self.t_ref
            self.lag = self.predict(t_c) + self.alpha * resid
            if dt > 0:
                self.rate = float(np.clip(self.rate + self.beta * resid / dt,
                                          -self.rate_limit, self.rate_limit))
            self.t_ref = float(t_c)
            self.confidence, self.last_update = conf, t_c
            self._outliers.clear()
            self.n_updates += 1
            self._refit(t_c, delay)
            return True
        # Ausreißer (außerhalb ±wide): nur nach drei untereinander stimmigen, sehr sicheren Messungen neu aufsetzen
        if conf >= self.reacquire_conf:
            if self._outliers and abs(delay - self._outliers[-1][1]) > self.tol:
                self._outliers.clear()
            self._outliers.append((t_c, delay))
            if len(self._outliers) >= 3 and t_c - self._outliers[0][0] >= self.relock_span:
                (t0, d0), (t1, d1) = self._outliers[0], self._outliers[-1]
                self.lag, self.t_ref = float(d1), float(t1)
                self.rate = float(np.clip((d1 - d0) / (t1 - t0), -self.rate_limit, self.rate_limit))
                self.confidence, self.last_update = conf, t_c
                self._outliers.clear()
                self.n_updates += 1
                self._hist = [(t0, d0), (t1, d1)]
                return True
        return False

    def _refit(self, t_c: float, delay: float) -> None:
        """Geradenanpassung über die letzten Messungen (Glättung, genaue Driftrate)."""
        self._hist.append((t_c, delay))
        cut = t_c - self.fit_span
        self._hist = [h for h in self._hist if h[0] >= cut]
        if len(self._hist) < 4:
            return
        t = np.array([h[0] for h in self._hist])
        d = np.array([h[1] for h in self._hist])
        tm = t.mean()
        if t.max() - t.min() < self.fit_min_span:
            return
        if self.robust and self.rate_limit == 0.0:
            core = d[np.abs(d - np.median(d)) <= self.tol]
            self.lag, self.rate, self.t_ref = float(core.mean()), 0.0, float(t_c)
            return
        slope, icpt = np.polyfit(t - tm, d, 1)
        slope = float(np.clip(slope, -self.rate_limit, self.rate_limit))
        self.rate = slope
        self.lag = float(icpt + slope * (t_c - tm))
        self.t_ref = float(t_c)
