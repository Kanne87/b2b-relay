"""Ausgangsstufe mit variabler Leserate (Varispeed, ohne Tonhöhenkorrektur).

``VarispeedReader`` liest das Programm P(T) mit einer fraktionalen Leseposition, die je
Ausgabe-Sample um die Rate ``r`` weiterläuft. Die Rate darf sich innerhalb eines Blocks linear
ändern; die Position ist die exakte Summe (geschlossene Form), daher gibt es weder Phasensprünge
noch Klicks an Blockgrenzen oder bei Ratenwechsel.

Position als ``(int, frac)``: der ganzzahlige Anteil wächst über Stunden unbegrenzt, der Bruchteil
bleibt in [0, 1). So bleibt die Genauigkeit unabhängig von der Laufzeit (float64 allein verlöre
nichts Hörbares, aber es gäbe keine feste Schranke).

Interpolation: fensterbewertete Sinc-Interpolation (Kaiser, 32 Stützstellen, Tabelle mit
32768 Phasen, nächste Phase: Positionsfehler < 0,00002 Sample). Bruchteil 0 liefert exakt den Eingangswert; der Frequenzgang ist
bis ca. 18 kHz auf < 0,01 dB flach und hängt kaum vom Bruchteil ab (kubisches Lagrange würde
bei ständig wanderndem Bruchteil die Höhen mit der Wanderfrequenz amplitudenmodulieren).

Schnittstelle (``render``, ``position``, ``margin``) ist so geschnitten, dass eine Keylock-/
Time-Stretch-Stufe (z. B. librubberband) denselben Platz einnehmen kann: sie bekommt eine
Lesefunktion ``read(t0, n) -> P(t0 .. t0+n)`` und eine Rate pro Block und liefert ``n`` Samples.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ReadFn = Callable[[int, int], np.ndarray]


class SincTable:
    """Polyphase-Tabelle: ``rows[m]`` sind die Gewichte für den Bruchteil ``m / phases``."""

    def __init__(self, half: int = 16, phases: int = 32768, beta: float = 8.0) -> None:
        self.half, self.phases = half, phases
        k = np.arange(-half + 1, half + 1, dtype=np.float64)           # Stützstellen -H+1 .. H
        f = (np.arange(phases + 1, dtype=np.float64) / phases)[:, None]
        x = k[None, :] - f                                              # Abstand zur Leseposition
        with np.errstate(divide="ignore", invalid="ignore"):
            # sin(pi (k - f)) = -(-1)^k sin(pi f): bei f = 0 exakt 0 (np.sinc liefert dort ~1e-16)
            s = -((-1.0) ** k)[None, :] * np.sin(np.pi * f) / (np.pi * x)
        s = np.where(x == 0.0, 1.0, s)
        win = np.i0(beta * np.sqrt(np.clip(1.0 - (x / half) ** 2, 0.0, None))) / np.i0(beta)
        rows = s * win
        rows /= rows.sum(axis=1, keepdims=True)
        rows[0] = 0.0
        rows[0, half - 1] = 1.0              # Bruchteil 0: exakt Identität
        rows[phases] = np.roll(rows[0], 1)   # Bruchteil 1 = um eine Stelle versetzt
        self.rows = rows
        self.k = k.astype(np.int64)


_TABLE: SincTable | None = None


def sinc_table() -> SincTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = SincTable()
    return _TABLE


class VarispeedReader:
    def __init__(self, start: float = 0.0) -> None:
        self._tab = sinc_table()
        i = int(np.floor(start))
        self.pos_int = i
        self.pos_frac = float(start - i)

    @property
    def position(self) -> float:
        """Programmzeit (Samples) des nächsten Ausgabe-Samples."""
        return self.pos_int + self.pos_frac

    @property
    def margin(self) -> int:
        """Samples, die die Stufe über die Leseposition hinaus (vorwärts) braucht."""
        return self._tab.half

    def advance_of(self, n: int, r0: float, r1: float) -> float:
        """Vorschub der Position über einen Block (Samples)."""
        return n * 0.5 * (r0 + r1)

    def offsets(self, n: int, r0: float, r1: float) -> np.ndarray:
        """Positionen der n+1 Gitterpunkte relativ zum Blockanfang: r0·i + (r1−r0)·i²/(2n)."""
        i = np.arange(n + 1, dtype=np.float64)
        return r0 * i + (r1 - r0) * i * i / (2.0 * n)

    def render(self, read: ReadFn, n: int, r0: float = 1.0, r1: float | None = None) -> np.ndarray:
        """``n`` Ausgabe-Samples; die Rate geht über den Block linear von ``r0`` nach ``r1``."""
        r1 = r0 if r1 is None else r1
        tab = self._tab
        H = tab.half
        off = self.offsets(n, r0, r1)
        base = self.pos_frac + off                      # klein: |base| < n·r + 1
        ip = np.floor(base[:-1])
        frac = base[:-1] - ip
        ip = ip.astype(np.int64)
        lo = self.pos_int + int(ip[0]) - H + 1
        hi = self.pos_int + int(ip[-1]) + H
        seg = np.asarray(read(lo, hi - lo + 1), dtype=np.float64)
        if r0 == 1.0 and r1 == 1.0:
            # Tempo 1: der Bruchteil bleibt im ganzen Block gleich → eine Faltung genügt
            m = int(self.pos_frac * tab.phases + 0.5)
            out = np.convolve(seg, tab.rows[m][::-1], mode="valid")[:n]
        else:
            m = (frac * tab.phases + 0.5).astype(np.int64)
            win = np.lib.stride_tricks.sliding_window_view(seg, 2 * H)     # (len-2H+1, 2H), ohne Kopie
            out = np.einsum("ik,ik->i", win[ip - int(ip[0])], tab.rows[m])
        total = self.pos_frac + float(off[-1])
        step = int(np.floor(total))
        self.pos_int += step
        self.pos_frac = total - step
        return out

