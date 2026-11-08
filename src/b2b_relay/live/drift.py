"""Verankerung der Programmzeitachse T an der Wanduhr (CLOCK_MONOTONIC).

Hintergrund (README, Grenzen 1b): Der Mixer misst jeden neuen Lag relativ zum Programm, das er mit
dem *geschätzten* Lag der vorigen Quelle zusammensetzt. Fehler der Driftrate summieren sich über die
Übergabekette; die Zeitachse wandert gegen das wahre Raster (Simulation bis 118 ms nach 30 Übergaben).
Im Live-Betrieb kommt ein zweiter Anteil dazu: die JACK-Uhr läuft gegen CLOCK_MONOTONIC.

Messgröße ``e`` (Sekunden): Abweichung des Live-Rands der Programmzeit von der Wanduhr-Sollkurve

    T_soll(t) = T_ref + (t_mono − mono_ref) − Σ ΔL_k

Die Wanduhr läuft mit Faktor 1; jede Übergabe ``k`` setzt den Live-Rand um den Lag-Zuwachs ΔL_k
(≈ RTT des Übernehmenden) zurück, das wird gutgeschrieben (``credit``). Alles andere - Lag-Drift der
aktiven Quelle innerhalb einer Haltezeit, Fehler der Driftrate, JACK-Uhr gegen CLOCK_MONOTONIC - ist
Achsendrift. ``e = (T_live − T_ref)/fs − (t_mono − mono_ref) + Σ ΔL_k/fs``.

Regelung: Das Mixer-Programm liegt zwischen den Quellen fest (Lags sind relativ), ein Verschieben der
Achse als Ganzes ändert das Audio nicht. Deshalb wird der Rest als **Umlabeln** der Zeitachse
(``RelayMixer.retime``) ausgeregelt, in kleinen Schritten (Steigung begrenzt, Totband), nicht als
Sprung. Die Varispeed-Stufe regelt weiter den Puffer B (sie kann die Lage der Achse nicht verändern,
nur die Ausgabeposition gegen den Live-Rand); die Rate steht im Status neben der Drift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftStatus:
    anchored: bool
    drift_ms: float             # e nach Korrektur (Regelgröße)
    raw_drift_ms: float         # e ohne bisherige Korrekturen (was die Achse ohne Eingriff getan hätte)
    corrected_ms: float         # Summe aller Umlabel-Schritte
    ppm_60s: float | None       # Steigung der Rohdrift über die letzten 60 s
    max_abs_ms: float           # größte |e| seit Verankerung (nach dem Einschwingen: 30 s)
    credits_ms: float           # Summe der gutgeschriebenen Lag-Zuwächse


class DriftMonitor:
    def __init__(self, fs: int, *, enable: bool = True, deadband_ms: float = 1.0,
                 slew_ms_per_s: float = 1.0, settle_s: float = 30.0) -> None:
        self.fs = fs
        self.enable = enable
        self.deadband = deadband_ms * 1e-3
        self.slew = slew_ms_per_s * 1e-3
        self.settle_s = settle_s
        self.anchored = False
        self._t_ref = 0.0           # Samples
        self._mono_ref = 0.0
        self._credit = 0.0          # Samples
        self._corrected = 0         # Samples (Summe der retime-Schritte)
        self._last_active: str | None = None
        self._last_lag_old: float = 0.0
        self._e = 0.0
        self._max_abs = 0.0
        self._hist: deque[tuple[float, float]] = deque()    # (mono, rohe e in s)

    def anchor(self, live_edge: int, mono: float) -> None:
        self._t_ref, self._mono_ref, self.anchored = float(live_edge), mono, True

    def credit_handover(self, lag_new: float, lag_old: float) -> None:
        """Lag-Zuwachs einer wirksam gewordenen Übergabe (Samples) gutschreiben."""
        self._credit += lag_new - lag_old

    def step(self, live_edge: int, mono: float, dt: float) -> int:
        """Einen Messpunkt verarbeiten; gibt den **Umlabel-Schritt** (Samples) zurück, den der Aufrufer
        am Mixer ausführen soll (0 = nichts). ``mono`` = Wanduhrzeit des Live-Rands (s)."""
        if not self.anchored:
            return 0
        e = ((live_edge - self._t_ref) + self._credit) / self.fs - (mono - self._mono_ref)
        self._e = e
        raw = e - self._corrected / self.fs
        self._hist.append((mono, raw))
        while self._hist and mono - self._hist[0][0] > 60.0:
            self._hist.popleft()
        if mono - self._mono_ref >= self.settle_s:
            self._max_abs = max(self._max_abs, abs(e))
        if not self.enable or abs(e) <= self.deadband:
            return 0
        step_max = max(1, int(self.slew * dt * self.fs))
        want = -int(round(e * self.fs))
        d = max(-step_max, min(step_max, want))
        if d:
            self._corrected += d
            self._e = e + d / self.fs
        return d

    def status(self) -> DriftStatus:
        if not self.anchored:
            return DriftStatus(False, 0.0, 0.0, 0.0, None, 0.0, 0.0)
        ppm = None
        if len(self._hist) >= 2 and self._hist[-1][0] - self._hist[0][0] >= 20.0:
            (t0, x0), (t1, x1) = self._hist[0], self._hist[-1]
            ppm = (x1 - x0) / (t1 - t0) * 1e6
        return DriftStatus(True, self._e * 1e3, (self._e - self._corrected / self.fs) * 1e3,
                           self._corrected / self.fs * 1e3, ppm, self._max_abs * 1e3,
                           self._credit / self.fs * 1e3)
