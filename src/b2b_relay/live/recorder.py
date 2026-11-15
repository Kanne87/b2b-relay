"""Segmentierter Mitschnitt der Programmausgabe (WAV + Trace), fehlertolerant.

Eine WAV-Datei trägt höchstens 4 GiB (32-Bit-Längenfelder); die Dauer-Aufnahme lief früher in diese Grenze
und der Worker blieb stehen (siehe README, "Aufnahme"). Deshalb gilt hier:

* Segmente von ``record_segment_s`` Sekunden (Wechsel an der Blockgrenze, kein Sample geht verloren oder
  doppelt); Dateinamen ``program-<Start>-<Nr>.wav`` / ``.trace.jsonl`` sind nach Name sortierbar.
* Jeder Fehler beim Öffnen/Schreiben wird abgefangen, als Ereignis ``recording_error`` gemeldet; das Segment wird
  geschlossen, nach ``RETRY_S`` beginnt ein neues. Der Aufrufer (Worker) sieht nie eine Ausnahme.
* Gesamtgröße der eigenen Dateien ≤ ``record_max_mb``: die ältesten Segmente werden gelöscht (nur Namen nach
  dem eigenen Muster, nie das aktuelle Segment).
* Optional nur mit live spielender Quelle; nach ``HOLD_S`` ohne Quelle endet das Segment.

Zeit rechnet der Recorder in Frames (Aufrufer übergibt ``k``), nicht in Wanduhrzeit.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import LiveConfig
from .wav import WavWriter

RETRY_S = 10.0          # nach einem Fehler so lange keine Aufnahme, dann neues Segment
HOLD_S = 2.0            # Nachlauf, nachdem keine Quelle mehr live ist
ENFORCE_S = 10.0        # Abstand der Platzprüfung während eines Segments
_NAME = re.compile(r"^program-.+\.(wav|trace\.jsonl)$")


class Recorder:
    def __init__(self, cfg: LiveConfig, fs: int, emit: Callable[..., None],
                 wav_factory: Callable[..., Any] = WavWriter) -> None:
        self.cfg, self.fs, self._emit = cfg, fs, emit
        self._factory = wav_factory
        self.dir = Path(cfg.record_dir) if cfg.record_dir else None
        self.enabled = self.dir is not None
        self._wav: Any = None
        self._trace: Any = None
        self.current: str | None = None
        self.trace_frames = 0           # Frames im aktuellen Segment
        self._seg_frames_max = max(1, int(cfg.record_segment_s * fs))
        self._stamp = ""
        self._seq = 0
        self._retry_in = 0              # Frames bis zum nächsten Versuch nach Fehler
        self._hold = 0                  # Frames Nachlauf ohne Quelle
        self._since_enforce = 0
        self.state = "idle" if self.enabled else "off"
        self.reason = "" if self.enabled else "B2B_RECORD_DIR nicht gesetzt"
        self.errors = 0
        self.deleted = 0
        self.last_label = ""
        self._stats = (0, 0)
        self._stats_at = 0.0

    # ------------------------------------------------------------------ Steuerung

    @property
    def active(self) -> bool:
        return self._wav is not None

    def write(self, left, right, k: int, live: bool, trace_lines: Callable[[int], list[str]] | None = None) -> None:
        """Einen Block aufnehmen (oder nicht). Wirft nie."""
        if not self.enabled:
            return
        try:
            self._write(left, right, k, live, trace_lines)
        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")

    def close(self) -> None:
        self._close_files()
        if self.enabled and self.state not in ("error",):
            self.state, self.reason = "idle", "beendet"

    # ------------------------------------------------------------------ intern

    def _write(self, left, right, k, live, trace_lines) -> None:
        if self._retry_in > 0:
            self._retry_in -= k
            if self._retry_in > 0:
                return
            self._retry_in = 0
        if live:
            self._hold = int(HOLD_S * self.fs)
        elif self._hold > 0:
            self._hold -= k
        if self.cfg.record_only_live and not live and self._hold <= 0:
            if self.active:
                self._close_files()
            self.state, self.reason = "waiting", "keine Quelle live"
            return
        if self.active and self.trace_frames >= self._seg_frames_max:
            self._close_files()                         # Segmentwechsel an der Blockgrenze
        if not self.active:
            self._open()
        self._wav.write((left, right))
        if trace_lines is not None:
            for line in trace_lines(self.trace_frames):
                self._trace.write(line + "\n")
        self.trace_frames += k
        self.state, self.reason = "recording", ""
        self._since_enforce += k
        if self._since_enforce >= ENFORCE_S * self.fs:
            self._since_enforce = 0
            self.enforce_limit()

    def _open(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self._stamp:
            self._stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        while True:
            self._seq += 1
            base = self.dir / f"program-{self._stamp}-{self._seq:04d}"
            if not base.with_suffix(".wav").exists():
                break
        self.enforce_limit()                            # Platz schaffen, bevor wir ein Segment beginnen
        self.last_label = str(base) + ".wav"
        self._wav = self._factory(self.last_label, self.fs, 2, self.cfg.record_format)
        self._trace = open(str(base) + ".trace.jsonl", "w", buffering=1)
        self.current = self.last_label
        self.trace_frames = 0
        self._since_enforce = 0

    def _close_files(self) -> None:
        w, t, self._wav, self._trace = self._wav, self._trace, None, None
        for f in (w, t):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self.current = None

    def _fail(self, why: str) -> None:
        self.errors += 1
        self._close_files()
        self._retry_in = int(RETRY_S * self.fs)
        self.state, self.reason = "error", why
        self._emit("recording_error", reason=why, retry_s=RETRY_S)

    # ------------------------------------------------------------------ Platz

    def _own_files(self) -> list[Path]:
        if self.dir is None or not self.dir.is_dir():
            return []
        return sorted(p for p in self.dir.iterdir() if _NAME.match(p.name) and p.is_file())

    def stats(self) -> tuple[int, int]:
        """(Anzahl Segmente, Gesamtgröße der eigenen Dateien in Bytes)."""
        files = self._own_files()
        total = 0
        for p in files:
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return sum(1 for p in files if p.name.endswith(".wav")), total

    def enforce_limit(self) -> None:
        limit = self.cfg.record_max_mb * 1024 * 1024
        if limit <= 0 or not self.enabled:
            return
        try:
            files = self._own_files()
            sizes = {}
            for p in files:
                try:
                    sizes[p] = p.stat().st_size
                except OSError:
                    sizes[p] = 0
            total = sum(sizes.values())
            keep = {Path(self.current), Path(self.current[:-4] + ".trace.jsonl")} if self.current else set()
            for p in files:                              # Namen sortieren nach Zeit: älteste zuerst
                if total <= limit:
                    break
                if p in keep:
                    continue
                try:
                    p.unlink()
                except OSError:
                    continue
                total -= sizes[p]
                self.deleted += 1
        except Exception as e:                           # Aufräumen darf nie stören
            self._emit("recording_error", reason=f"Aufräumen: {type(e).__name__}: {e}", retry_s=0)

    def status(self) -> dict:
        now = time.monotonic()
        if self.enabled and now >= self._stats_at:      # Verzeichnis höchstens alle 2 s lesen
            self._stats, self._stats_at = self.stats(), now + 2.0
        n, total = self._stats
        return {"file": self.current, "state": self.state, "reason": self.reason,
                "segments": n, "bytes": total, "errors": self.errors, "deleted": self.deleted,
                "segment_s": self.cfg.record_segment_s, "max_mb": self.cfg.record_max_mb,
                "only_live": self.cfg.record_only_live}
