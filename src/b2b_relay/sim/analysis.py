"""Messungen am Programm-Audio: Kick-Onsets gegen ein Referenz-Raster, Lücken, Sample-Sprünge."""

from __future__ import annotations

import numpy as np

from ..timeline import frac_read

try:
    from scipy.signal import butter, sosfiltfilt
except ImportError as e:    # pragma: no cover
    raise ImportError("Die Simulation braucht scipy: pip install -e .[sim]") from e

KICK_BAND_HZ = 140.0


def _lowpass(x: np.ndarray, fs: int) -> np.ndarray:
    sos = butter(4, KICK_BAND_HZ, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def onset_time(x_lp: np.ndarray, fs: int, guess: float, search_s: float = 0.03,
               rise_frac: float = 0.35) -> float | None:
    """Onset eines Kicks im tiefpassgefilterten Signal ``x_lp`` nahe Index ``guess``.

    Hüllkurve = geglättetes |x|; Onset = erster Durchgang der Hüllkurve durch
    ``base + rise_frac · (peak − base)`` (linear interpoliert), ``base`` = Hüllkurven-Minimum in
    den 15 ms vor dem Suchfenster, ``peak`` = Maximum in den 50 ms nach dem Fenster-Beginn.
    """
    k = max(3, int(0.0015 * fs))
    env = np.convolve(np.abs(x_lp), np.hanning(k) / np.hanning(k).sum(), mode="same")
    a = int(guess - search_s * fs)
    b = int(guess + search_s * fs)
    if a < int(0.02 * fs) or b + int(0.06 * fs) >= len(env):
        return None
    base = float(env[a - int(0.015 * fs):a].min())
    peak = float(env[a:b + int(0.05 * fs)].max())
    if peak - base < 1e-4:
        return None
    thr = base + rise_frac * (peak - base)
    seg = env[a:b + int(0.05 * fs)]
    above = np.flatnonzero(seg >= thr)
    if len(above) == 0 or above[0] == 0:
        return None
    i = above[0]
    y0, y1 = seg[i - 1], seg[i]
    frac = (thr - y0) / (y1 - y0) if y1 != y0 else 0.0
    return a + i - 1 + frac


def kick_phase_errors(prog: np.ndarray, ref_fn, kicks: np.ndarray, fs: int,
                      loss_intervals: np.ndarray | None = None,
                      pad_s: float = 0.35) -> list[dict]:
    """Phasenfehler des Programms gegen die Referenz je Kick.

    ``prog``: Programm, Index = Programmzeit T. ``ref_fn(positions)``: Referenz-Signal (ideales
    Programm der Quelle der Kicks) an Programmzeiten T. ``kicks``: erwartete Kick-Zeiten (T,
    Samples, Bruchteil). Fehler = Onset(Programm) − Onset(Referenz), positiv = Programm zu spät.
    Kicks, deren Fenster einen Paketverlust berührt, werden markiert und nicht gewertet.
    """
    out = []
    pad = int(pad_s * fs)
    for tk in kicks:
        i0 = int(np.floor(tk)) - pad
        n = 2 * pad
        if i0 < 0 or i0 + n > len(prog):
            continue
        rec = {"t": float(tk), "ok": False, "err_ms": None}
        if loss_intervals is not None and len(loss_intervals):
            lo, hi = tk - 0.06 * fs, tk + 0.12 * fs
            if np.any((loss_intervals[:, 1] > lo) & (loss_intervals[:, 0] < hi)):
                rec["lost"] = True
                out.append(rec)
                continue
        p = _lowpass(prog[i0:i0 + n].astype(np.float64), fs)
        r = _lowpass(ref_fn(np.arange(i0, i0 + n, dtype=np.float64)), fs)
        guess = tk - i0
        tp = onset_time(p, fs, guess)
        tr = onset_time(r, fs, guess)
        if tp is not None and tr is not None:
            rec["ok"] = True
            rec["err_ms"] = float((tp - tr) / fs * 1e3)
        out.append(rec)
    return out


def find_gaps(x: np.ndarray, fs: int, min_ms: float = 5.0, thr: float = 2e-5) -> list[tuple[int, int]]:
    """Läufe digitaler Stille (|x| < thr) von mindestens ``min_ms``: [(start, länge)]."""
    quiet = np.abs(x) < thr
    if not quiet.any():
        return []
    d = np.diff(np.concatenate(([0], quiet.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    min_len = int(min_ms * 1e-3 * fs)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends) if e - s >= min_len]


def max_sample_jump(x: np.ndarray, exclude: np.ndarray | None = None) -> float:
    """Größter Betrag der Differenz benachbarter Samples (ohne markierte Bereiche)."""
    if len(x) < 2:
        return 0.0
    d = np.abs(np.diff(x.astype(np.float64)))
    if exclude is not None and len(exclude):
        d = np.where(exclude[1:] | exclude[:-1], 0.0, d)
    return float(d.max())
