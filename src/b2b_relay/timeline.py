"""Programm-Zeitachse: Ringpuffer mit absolutem Sample-Index und Lesen mit Bruchteil-Versatz.

Zwei Indexräume, die man nicht verwechseln darf:

* Wanduhr ``w``: Index, unter dem ein Sample einer Quelle beim Mixer *ankommt* (zählt ab dem
  ersten ``process``-Aufruf, immer +n je Block).
* Programmzeit ``T``: Index im Programm P(T). Eine Quelle X liefert den Programminhalt T zur
  Wanduhrzeit ``w = T + lag_X``.
"""

from __future__ import annotations

import numpy as np


class SampleRing:
    """Ringpuffer mit fortlaufendem absolutem Index.

    ``write`` hängt Samples an (Index ``end`` wächst). ``read`` liefert beliebige absolute
    Bereiche; was nie geschrieben oder bereits überschrieben wurde, kommt als Nullen zurück.
    ``start`` ist der absolute Index des ersten Samples: Der Mixer legt den Ring einer Quelle mit
    der Wanduhr ihrer Anmeldung an, damit Ring-Index und Wanduhr gleich sind (S3904).
    """

    def __init__(self, capacity: int, start: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity muss > 0 sein")
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self.end = int(start)  # absoluter Index des nächsten zu schreibenden Samples

    @property
    def start(self) -> int:
        """Ältester noch lesbarer absoluter Index (davor Liegendes kommt als Null)."""
        return max(0, self.end - self.capacity)

    def write(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32)
        n = block.shape[0]
        if n >= self.capacity:
            block = block[n - self.capacity:]
            self.end += n - self.capacity
            n = self.capacity
        pos = self.end % self.capacity
        first = min(n, self.capacity - pos)
        self._buf[pos:pos + first] = block[:first]
        if first < n:
            self._buf[:n - first] = block[first:]
        self.end += n

    def overwrite(self, start: int, block: np.ndarray) -> None:
        """Bereits geschriebenen Bereich [start, start+len) ersetzen (nur, was noch im Ring liegt)."""
        block = np.asarray(block, dtype=np.float32)
        lo = max(start, self.start)
        hi = min(start + len(block), self.end)
        if hi <= lo:
            return
        idx = np.arange(lo, hi) % self.capacity
        self._buf[idx] = block[lo - start:hi - start]

    def read(self, start: int, n: int) -> np.ndarray:
        """Absoluter Bereich [start, start+n) als float32; Unverfügbares ist 0."""
        out = np.zeros(n, dtype=np.float32)
        lo = max(start, self.start)
        hi = min(start + n, self.end)
        if hi <= lo:
            return out
        m = hi - lo
        pos = lo % self.capacity
        first = min(m, self.capacity - pos)
        o = lo - start
        out[o:o + first] = self._buf[pos:pos + first]
        if first < m:
            out[o + first:o + m] = self._buf[:m - first]
        return out

    def read_frac(self, positions: np.ndarray) -> np.ndarray:
        """Kubisch interpolierte Werte an (aufsteigenden) Bruchteil-Positionen."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.size == 0:
            return np.zeros(0, dtype=np.float64)
        i0 = int(np.floor(positions[0])) - 1
        i1 = int(np.floor(positions[-1])) + 3
        seg = self.read(i0, i1 - i0 + 1)
        return frac_read(seg, positions - i0)


def frac_read(buf: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """4-Punkt-Lagrange-Interpolation von ``buf`` an Positionen ``pos`` (Index im Array).

    Positionen ohne vollständige Nachbarschaft (``<1`` oder ``>len-3``) ergeben 0.
    Bei ganzzahliger Position ist das Ergebnis exakt der Arraywert. (Farrow-Form.)
    """
    pos = np.asarray(pos, dtype=np.float64)
    if buf.shape[0] < 4:
        return np.zeros(pos.shape, dtype=np.float64)
    i = np.floor(pos).astype(np.intp)
    f = pos - i
    ok = (i >= 1) & (i <= buf.shape[0] - 3)
    all_ok = bool(ok.all())
    if not all_ok:
        i = np.clip(i, 1, buf.shape[0] - 3)
    a = buf[i - 1].astype(np.float64)
    b = buf[i].astype(np.float64)
    c = buf[i + 1].astype(np.float64)
    d = buf[i + 2].astype(np.float64)
    c1 = c - b * 0.5 - a * (1.0 / 3.0) - d * (1.0 / 6.0)
    c2 = (a + c) * 0.5 - b
    c3 = (b - c) * 0.5 + (d - a) * (1.0 / 6.0)
    y = b + f * (c1 + f * (c2 + f * c3))
    return y if all_ok else np.where(ok, y, 0.0)
