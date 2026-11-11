"""Verschleierung kurzer Aussetzer im Feed einer Quelle (Paketverlust, Jitterpuffer-Unterlauf).

JackTrip (Jitterpuffer-Strategie 1, ``-q auto``) gibt für ein verlorenes oder zu spät gekommenes Paket
exakte Nullen aus: ein Lauf von 256 Frames (ein JACK-Takt) oder einem Vielfachen. Im Programm ist das
eine echte Lücke (digitale Stille, 5,3 ms je Paket). Musik enthält keine Läufe exakter Nullen mitten im
Signal; ein solcher Lauf zwischen zwei Signalstücken ist deshalb ein Aussetzer der Strecke.

``GapConcealer`` sieht jeden neu geschriebenen Block an. Ist ein Lauf exakter Nullen
(``min_len`` … ``max_len`` Samples) auf beiden Seiten von Signal umgeben, ersetzt er ihn im Ring durch das
Signal, das ``P = L + fade`` Samples davor lag (Wellenform-Ersatz), mit Kosinus-Überblendung über ``fade``
Samples auf beiden Seiten - kein Knacks, keine Stille. Das geht nachträglich, weil der Mixer jede Quelle
erst rund eine Puffer-Länge (≥ 1 s) nach dem Eintreffen ausgibt. Längere Läufe (Quelle verstummt) bleiben
unverändert; dafür gibt es den Fallback des Mixers.
"""

from __future__ import annotations

import numpy as np

from .timeline import SampleRing


class GapConcealer:
    def __init__(self, max_len: int, min_len: int = 16, fade: int = 64) -> None:
        self.max_len, self.min_len, self.fade = int(max_len), int(min_len), int(fade)
        self._open: int | None = None           # Beginn eines noch offenen Nulllaufs (absolut)
        self._pending: list[tuple[int, int]] = []
        self._first_signal: int | None = None
        self.count = 0                          # verschleierte Aussetzer
        self.samples = 0                        # davon Samples

    def after_write(self, ring: SampleRing, w0: int, blk: np.ndarray) -> None:
        """``blk`` ist eben unter [w0, w0+len) in ``ring`` geschrieben worden."""
        n = len(blk)
        if n == 0:
            return
        z = blk == 0.0
        if self._first_signal is None:
            nz = np.flatnonzero(~z)
            if len(nz) == 0:
                return
            self._first_signal = w0 + int(nz[0])
            self._open = None
        # Grenzen der Nullläufe im Block (Wechsel 0→1 bzw. 1→0)
        d = np.diff(z.astype(np.int8))
        starts = list(np.flatnonzero(d == 1) + 1)
        ends = list(np.flatnonzero(d == -1) + 1)
        if z[0]:
            starts.insert(0, 0)
        elif self._open is not None:            # offener Lauf endete genau an der Blockgrenze
            self._close(self._open, w0)
            self._open = None
        if z[-1]:
            ends.append(n)
        for a, b in zip(starts, ends):
            a_abs, b_abs = w0 + int(a), w0 + int(b)
            if a == 0 and self._open is not None:
                a_abs = self._open
            self._open = None
            if b == n:                           # Lauf reicht bis ans Blockende: noch offen
                self._open = a_abs
                continue
            self._close(a_abs, b_abs)
        while self._pending and self._pending[0][1] + self.fade <= ring.end:
            self._conceal(ring, *self._pending.pop(0))

    def _close(self, a: int, b: int) -> None:
        if a > self._first_signal and self.min_len <= b - a <= self.max_len:
            self._pending.append((a, b))

    def _conceal(self, ring: SampleRing, a: int, b: int) -> None:
        m, L = self.fade, b - a
        p = L + m
        lo = a - m
        if lo - p < ring.start:
            return
        orig = ring.read(lo, L + 2 * m).astype(np.float64)
        copy = ring.read(lo - p, L + 2 * m).astype(np.float64)
        u = 0.5 - 0.5 * np.cos(np.pi * (np.arange(m) + 0.5) / m)
        new = copy.copy()
        new[:m] = orig[:m] * (1.0 - u) + copy[:m] * u
        new[m + L:] = copy[m + L:] * (1.0 - u) + orig[m + L:] * u
        ring.overwrite(lo, new.astype(np.float32))
        self.count += 1
        self.samples += L
