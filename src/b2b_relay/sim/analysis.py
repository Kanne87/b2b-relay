"""Messungen am Programm-Audio: Kick-Onsets gegen ein Referenz-Raster, Lücken, Sample-Sprünge."""

from __future__ import annotations

import numpy as np

from ..timeline import frac_read

try:
    from scipy.signal import butter, sosfiltfilt
except ImportError as e:    # pragma: no cover
    raise ImportError("Die Simulation braucht scipy: pip install -e .[sim]") from e

KICK_BAND_HZ = 80.0


def _lowpass(x: np.ndarray, fs: int) -> np.ndarray:
    sos = butter(6, KICK_BAND_HZ, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def _envelope(x_lp: np.ndarray, fs: int) -> np.ndarray:
    k = max(3, int(0.0015 * fs))
    win = np.hanning(k + 2)[1:-1]
    return np.convolve(np.abs(x_lp), win / win.sum(), mode="same")


def onset_shift(prog_lp: np.ndarray, ref_lp: np.ndarray, fs: int, guess: float,
                max_shift_s: float = 0.015) -> float | None:
    """Verschiebung (Samples) des Kick-Onsets im Programm gegen die Referenz, positiv = später.

    Kick-Onset-Abgleich: In einem Fenster von 30 ms vor bis 100 ms nach dem erwarteten Kick
    (Raster) werden die Energiehüllkurven (geglättetes |x| des 80-Hz-Tiefpasses) von Programm und
    Referenz gegeneinander verschoben; die Lage des Korrelationsmaximums (Parabel-Verfeinerung)
    ist die Onset-Verschiebung. Gegen eine bloße Schwellen- oder Steilheitssuche ist das robust,
    weil Bass-Welligkeit und Abklingen des vorigen Kicks in beiden Hüllkurven gleich vorkommen.
    """
    a = int(guess - 0.03 * fs)
    b = int(guess + 0.10 * fs)
    m = int(max_shift_s * fs)
    if a - m < 0 or b + m >= len(prog_lp):
        return None
    ep = _envelope(prog_lp, fs)
    er = _envelope(ref_lp, fs)
    r = er[a:b]
    r = r - r.mean()
    if float(np.abs(r).max()) < 1e-4:
        return None
    cs = np.empty(2 * m + 1)
    for j, sft in enumerate(range(-m, m + 1)):
        q = ep[a + sft:b + sft]
        q = q - q.mean()
        cs[j] = float(np.dot(q, r)) / (np.linalg.norm(q) * np.linalg.norm(r) + 1e-30)
    k = int(np.argmax(cs))
    if cs[k] < 0.5:
        return None
    off = 0.0
    if 0 < k < len(cs) - 1:
        den = cs[k - 1] - 2 * cs[k] + cs[k + 1]
        off = 0.5 * (cs[k - 1] - cs[k + 1]) / den if den != 0 else 0.0
    return float(k - m + off)


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
        sh = onset_shift(p, r, fs, tk - i0)
        if sh is not None:
            rec["ok"] = True
            rec["err_ms"] = float(sh / fs * 1e3)
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
