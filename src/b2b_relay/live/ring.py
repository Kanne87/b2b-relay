"""Lock-freier Ringpuffer (ein Schreiber, ein Leser) für den JACK-Process-Callback.

Entwurf (Abnahme "Callback ohne Allokation, ohne Lock, ohne numpy-Neuanlage"):

* Der Speicher (``channels × capacity`` float32) wird einmal im Konstruktor angelegt.
* ``write``/``read`` kopieren per ``np.copyto`` zwischen Teilansichten (Slices) des Puffers und den
  JACK-Port-Arrays. Das erzeugt höchstens kleine View-Objekte, nie einen neuen Datenpuffer
  (``tests/test_live_ring.py`` prüft das mit ``tracemalloc``).
* Es gibt kein ``Lock``. Jede Seite schreibt genau einen Zähler: der Schreiber ``_w``, der Leser
  ``_r`` (monoton wachsende Python-ints, unter dem GIL atomar zu lesen/schreiben). Der Schreiber
  legt erst die Daten ab und erhöht dann ``_w``; der Leser liest ``_w``, dann die Daten, dann erhöht er
  ``_r``. Die Zähler sind absolute Frame-Indizes, nie zurückgesetzt (Python-ints laufen nicht über).
* Läuft der Puffer voll (Leser zu langsam), werden die *neuen* Frames verworfen und gezählt
  (``dropped``), statt den Leser zu überholen. Ist er leer, liefert ``read`` nur, was da ist; der Rest
  des Zielarrays bleibt unberührt (``read_padded`` füllt ihn mit Stille) und ``short`` zählt mit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class AudioRing:
    def __init__(self, channels: int, capacity: int) -> None:
        if channels < 1 or capacity < 1:
            raise ValueError("channels und capacity müssen >= 1 sein")
        self.channels = channels
        self.capacity = capacity
        self._buf = np.zeros((channels, capacity), dtype=np.float32)
        self._w = 0            # nur der Schreiber ändert ihn
        self._r = 0            # nur der Leser ändert ihn
        self.dropped = 0       # Schreiber: verworfene Frames (Überlauf)
        self.short = 0         # Leser: Aufrufe mit zu wenig Daten

    @property
    def available(self) -> int:
        """Lesbare Frames (vom Leser aus gesehen)."""
        return self._w - self._r

    @property
    def free(self) -> int:
        """Freier Platz (vom Schreiber aus gesehen)."""
        return self.capacity - (self._w - self._r)

    @property
    def written(self) -> int:
        return self._w

    @property
    def consumed(self) -> int:
        return self._r

    def write(self, srcs: Sequence[np.ndarray], n: int) -> int:
        """``n`` Frames aus ``srcs`` (ein Array je Kanal) anhängen; gibt die geschriebene Zahl zurück."""
        free = self.capacity - (self._w - self._r)
        m = n if n <= free else free
        if m < n:
            self.dropped += n - m
        if m <= 0:
            return 0
        pos = self._w % self.capacity
        first = self.capacity - pos
        if first > m:
            first = m
        buf = self._buf
        for ch in range(self.channels):
            src = srcs[ch]
            np.copyto(buf[ch, pos:pos + first], src[:first])
            if first < m:
                np.copyto(buf[ch, :m - first], src[first:m])
        self._w += m
        return m

    def read(self, dsts: Sequence[np.ndarray], n: int) -> int:
        """Bis zu ``n`` Frames in ``dsts`` kopieren; gibt die gelesene Zahl zurück."""
        avail = self._w - self._r
        m = n if n <= avail else avail
        if m < n:
            self.short += 1
        if m <= 0:
            return 0
        pos = self._r % self.capacity
        first = self.capacity - pos
        if first > m:
            first = m
        buf = self._buf
        for ch in range(self.channels):
            dst = dsts[ch]
            np.copyto(dst[:first], buf[ch, pos:pos + first])
            if first < m:
                np.copyto(dst[first:m], buf[ch, :m - first])
        self._r += m
        return m

    def read_padded(self, dsts: Sequence[np.ndarray], n: int) -> int:
        """Wie ``read``, der Rest bis ``n`` wird mit Stille gefüllt (kein neuer Speicher)."""
        m = self.read(dsts, n)
        if m < n:
            for d in dsts:
                d[m:n] = 0.0
        return m

    def skip(self, n: int) -> int:
        """Leserseitig ``n`` Frames verwerfen (ohne Kopie)."""
        m = min(n, self._w - self._r)
        if m > 0:
            self._r += m
        return max(m, 0)
