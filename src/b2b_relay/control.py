"""Puffer-Regelung des elastischen Ausgangs.

Modell: ``B = Live-Rand − Ausgabeposition`` (Sekunden Audio, das angekommen, aber noch nicht
ausgegeben ist). Liest der Ausgang die Zeitachse mit Rate ``r = 1 + dev``, ändert sich B um
``−dev`` je Sekunde: ``dev < 0`` (langsamer) füllt auf, ``dev > 0`` (schneller) leert. Jede
Übergabe senkt B sprunghaft um den Lag-Zuwachs.

Der Regler führt ``dev`` so, dass B im Totband um ``target`` bleibt:

* Sollwert ``dev*`` folgt der Bremsparabel ``sqrt(slew · |e|)`` (e = Abweichung vom Ruhepunkt),
  gedeckelt auf ``±max_dev``. Er läuft damit mit endlicher Steigung auf null aus, bevor B den
  Ruhepunkt erreicht (kein Überschwingen).
* ``dev`` folgt dem Sollwert nur mit höchstens ``slew`` je Sekunde (weiche Rampen, kein Ruck
  im Tempo).
* Innerhalb des Totbands (±``deadband``) ist der Sollwert 0 (keine Schwingung um den Zielwert);
  einmal eingegriffen, führt der Regler B bis ``deadband/2`` ans Ziel (Hysterese).
"""

from __future__ import annotations

import math


class BufferController:
    """Totband mit Hysterese: Der Regler greift erst bei ``|B − Ziel| > deadband_s`` ein und führt
    B dann bis ``deadband_s / 2`` an das Ziel heran (nicht nur bis an den Rand des Totbands)."""

    def __init__(self, target_s: float, deadband_s: float, max_dev: float, slew: float) -> None:
        if target_s <= 0 or deadband_s < 0 or max_dev <= 0 or slew <= 0:
            raise ValueError("Regler-Parameter müssen positiv sein")
        self.target_s = float(target_s)
        self.deadband_s = float(deadband_s)
        self.rest_s = 0.5 * self.deadband_s     # Restabweichung, bei der der Regler ausrastet
        self.max_dev = float(max_dev)
        self.slew = float(slew)
        self.dev = 0.0
        self.engaged = False

    def _law(self, buffer_s: float, dev: float, engaged: bool, dt: float) -> tuple[float, bool]:
        e = buffer_s - self.target_s
        if engaged and abs(e) <= self.rest_s:
            engaged = False
        elif not engaged and abs(e) > self.deadband_s:
            engaged = True
        if engaged:
            v = min(self.max_dev, math.sqrt(self.slew * (abs(e) - self.rest_s)))
            want = math.copysign(v, e)          # B zu groß → schneller (dev > 0)
        else:
            want = 0.0
        lim = self.slew * dt
        return dev + max(-lim, min(lim, want - dev)), engaged

    def step(self, buffer_s: float, dt: float) -> float:
        """Einen Zeitschritt ``dt`` (s) weiter; liefert das neue ``dev`` (= r − 1)."""
        self.dev, self.engaged = self._law(buffer_s, self.dev, self.engaged, dt)
        return self.dev

    def reach_limits(self) -> tuple[float, float]:
        """Puffer, bis zu dem der Regler von unten / von oben her führt."""
        return self.target_s - self.rest_s, self.target_s + self.rest_s

    def time_to(self, buffer_s: float, goal_s: float, horizon_s: float = 7200.0,
                dt: float = 0.05) -> float | None:
        """Sekunden, bis der Regler B von ``buffer_s`` auf ``goal_s`` geführt hat (Vorwärts-
        rechnung mit demselben Regelgesetz und dem aktuellen Reglerzustand, ohne weitere
        Übergaben). 0, wenn schon erreicht; ``None``, wenn der Regler B nie so weit führt (Ziel
        hinter dem Ruhepunkt) oder der Horizont nicht reicht."""
        up = goal_s > buffer_s
        if (up and buffer_s >= goal_s) or (not up and buffer_s <= goal_s):
            return 0.0
        lo, hi = self.reach_limits()
        if (up and goal_s > lo + 1e-9) or (not up and goal_s < hi - 1e-9):
            return None
        b, d, eng = float(buffer_s), self.dev, self.engaged
        t = 0.0
        while t < horizon_s:
            d, eng = self._law(b, d, eng, dt)
            nb = b - d * dt
            t += dt
            if (up and nb >= goal_s) or (not up and nb <= goal_s):
                frac = (goal_s - b) / (nb - b) if nb != b else 1.0
                return t - dt + frac * dt
            b = nb
        return None
