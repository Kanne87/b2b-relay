"""Nachbildung der Upload-Strecke des Integrationstests ohne Docker: ``tc netem`` (Client-Egress) +
JackTrip-Empfang am Hub (``UdpDataProtocol`` + ``JitterBuffer``, Strategie 1, Auto-Queue).

Quelle der Regeln: JackTrip v2.7.2 (``src/JitterBuffer.cpp``, ``src/UdpDataProtocol.cpp``), netem
``sch_netem.c`` (``rate``-Zweig). Abgebildet ist, was die Abtastwerte am Hub-Ausgang ``<dj>:receive_N``
verändert; Zeit in Frames (ein Paket = ein JACK-Takt = ``slot`` Frames).

* **netem** mit ``rate``: Sendezeit ``T_n = max(T_{n-1}, t_n + X_n)`` (Reihenfolge bleibt), ``X_n`` normal
  verteilt (``delay``, ``jitter``); ``loss`` verwirft, ``reorder`` (mit Korrelation) schickt ein Paket ohne
  Verzögerung vorweg.
* **UdpDataProtocol**: Pakete mit älterer Folgenummer werden verworfen; eine Lücke in den Folgenummern wird als
  Verlust (``lostLen``) an den Puffer gemeldet.
* **JitterBuffer** (Strategie 1): Unterlauf beim Lesen → Nullen (der Lesezeiger läuft weiter; kommt das Paket
  danach, setzt ``insertSlot`` den Lesezeiger höchstens einen Slot zurück = Nullen *eingefügt*, Lag +1 Slot);
  Überlauf beim Schreiben → ein Slot verworfen (Lag −1 Slot); Verlust → Nullen an Stelle des Inhalts.
  Auto-Queue: ``mMaxLatency`` folgt ``mAutoQueueCorr`` alle 2 s (Ausgabe „AutoQueue: n“ im Server-Log).
"""

from __future__ import annotations

import math

import numpy as np


class JitterBuffer:
    """JackTrip ``JitterBuffer`` mit Strategie 1, Einheiten Frames (ein Kanal genügt: alle Kanäle teilen
    Positionen und Entscheidungen)."""

    def __init__(self, slot: int = 256, qlen: int = -500, fs: int = 48000, channels: int = 2) -> None:
        self.slot = slot
        self.fs = fs
        self.total = 2 * fs                         # 2 s Ring
        self.ring = np.zeros((channels, self.total), np.float32)
        if qlen > 0:
            self.max_latency = qlen * slot
            self.auto = 0
        else:
            self.max_latency = 3 * slot
            self.auto = 1
        self.underrun_inc_tol = -10 * slot
        self.corr_inc_tol = 100 * self.max_latency
        self.write_pos = self.max_latency
        self.read_pos = 0
        self.level_down_rate = min(256, slot) / (5.0 * fs) * slot
        self.overflow_drop_step = slot              # Strategie 1
        self.level_cur = float(self.max_latency)
        self.min_level_thr = 1.9 * slot
        self.last_corr_counter = 0
        self.active = False
        self.in_slot = slot
        self.autoq_corr = 2.0 * slot
        self.autoq_factor = 1.0 / -qlen if qlen < 0 else 1.0 / 500
        self.autoq_rate = slot * 0.5
        self.autoq_rate_min = slot * 0.0005
        self.autoq_rate_decay = 1.0 - min(slot * 1.2e-6, 0.0005)
        self.queue_log: list[tuple[int, int]] = []  # (Lesezähler, Queue in Slots) bei jeder Änderung
        self.reads = 0
        # Zähler
        self.underrun_frames = 0
        self.overflows = 0
        self.lost_packets = 0

    def _corr_factor(self) -> float:
        return 500.0 / max(500, self.last_corr_counter)

    def insert(self, data: np.ndarray | None, lost_len: int = 0) -> None:
        n = self.slot
        if not self.active:
            self.active = True
        if self.max_latency < n + self.slot:
            self.max_latency = n + self.slot
        if lost_len > 0:
            self._packet_loss(lost_len)
        available = self.write_pos - self.read_pos
        delta = 0
        if available < -10 * self.max_latency:
            delta = available
            self.level_cur = n
        elif available + n > self.max_latency:
            delta = self.overflow_drop_step
            self.overflows += 1
            self.level_cur = self.max_latency
        elif available < 0 and self.level_cur < max(self.in_slot + self.min_level_thr,
                                                    self.max_latency - self.underrun_inc_tol
                                                    - 2 * self.slot * self._corr_factor()):
            delta = -min(-available, self.slot)
        elif self.level_cur < self.max_latency - self.corr_inc_tol - 6 * self.slot * self._corr_factor():
            delta = -self.slot
        if delta:
            self.read_pos += delta
            self.last_corr_counter = 0
        else:
            self.last_corr_counter += 1
        self._write(data if data is not None else None, n)

    def _write(self, data: np.ndarray | None, n: int) -> None:
        w = self.write_pos % self.total
        idx = (w + np.arange(n)) % self.total
        if data is None:
            self.ring[:, idx] = 0.0
        else:
            self.ring[:, idx] = data
        self.write_pos += n

    def _packet_loss(self, lost_len: int) -> None:
        available = self.write_pos - self.read_pos
        delta = min(available + self.in_slot + lost_len - self.max_latency, lost_len)
        if delta > 0:
            lost_len -= delta
            self.level_cur = self.max_latency
            self.last_corr_counter = 0
        elif self.slot < available + lost_len:          # Strategie 0/1: mOverflowDecTolerance > mMaxLatency
            delta = min(lost_len, self.slot)
            lost_len -= delta
            self.level_cur -= delta
            self.last_corr_counter = 0
        if lost_len > 0:
            self._write(None, lost_len)

    def read(self) -> np.ndarray:
        n = self.slot
        out = np.zeros((self.ring.shape[0], n), np.float32)
        if not self.active:
            return out
        available = self.write_pos - self.read_pos
        if available < self.level_cur:
            self.level_cur = max(float(available), self.level_cur - self.level_down_rate)
        else:
            self.level_cur = float(available)
        if available + self.autoq_corr - self.level_cur < 0:
            self.autoq_corr += self.autoq_rate
        elif self.in_slot + self.slot < self.autoq_corr:
            self.autoq_corr -= self.autoq_rate * self.autoq_factor
        if self.autoq_rate > self.autoq_rate_min:
            self.autoq_rate *= self.autoq_rate_decay
        if self.auto:
            pps = self.fs // self.slot
            if 2 * pps == self.auto % (4 * pps):
                k = 1.0 + 1e-5 / self.autoq_factor
                if 12 * pps > self.auto or abs(self.autoq_corr * k - self.max_latency + self.slot / 2) > 0.6 * self.slot:
                    self.max_latency = int(self.slot * math.ceil(self.autoq_corr * k / self.slot))
                    self.queue_log.append((self.reads, self.max_latency // self.slot))
            self.auto += 1
        read_len = max(0, min(available, n))
        if read_len:
            idx = (self.read_pos % self.total + np.arange(read_len)) % self.total
            out[:, :read_len] = self.ring[:, idx]
        if read_len < n:
            self.underrun_frames += n - read_len
        self.read_pos += n
        self.reads += 1
        return out


class NetemUplink:
    """Client-Egress mit ``tc netem delay D J distribution normal [loss L] [reorder R C] rate 1000mbit`` und
    JackTrip-Empfang am Hub. ``push`` nimmt einen Takt des Clients (Stereo-Block), ``pull`` liefert, was der
    Hub im selben Takt an ``receive_N`` ausgibt."""

    def __init__(self, delay_ms: float, jitter_ms: float = 0.0, loss: float = 0.0, reorder: float = 0.0,
                 reorder_corr: float = 0.0, qlen: int = -500, seed: int = 0, slot: int = 256,
                 fs: int = 48000) -> None:
        self.slot, self.fs = slot, fs
        self.delay = delay_ms * 1e-3 * fs
        self.jitter = jitter_ms * 1e-3 * fs
        self.loss, self.reorder, self.reorder_corr = loss, reorder, reorder_corr
        self.rng = np.random.default_rng(seed)
        self.jb = JitterBuffer(slot, qlen, fs)
        self.t = 0                              # Takt-Zähler (Frames)
        self.seq = 0
        self.last_send = -1e18                  # Sendezeit des letzten verzögerten Pakets (netem rate)
        self.inflight: list[tuple[float, int, np.ndarray]] = []
        self.last_seq: int | None = None        # UdpDataProtocol (None = mInitialState)
        self.out_of_order = 0
        self._ooo_run = 0
        self.lost = 0
        self._last_reorder_u = 1.0

    def _corr(self, u: float) -> float:
        # netem get_crandom: value = (1−ρ)·u + ρ·last
        v = (1.0 - self.reorder_corr) * u + self.reorder_corr * self._last_reorder_u
        self._last_reorder_u = v
        return v

    def push(self, l: np.ndarray, r: np.ndarray) -> None:
        now = float(self.t)
        seq = self.seq
        self.seq += 1
        if self.loss and self.rng.random() < self.loss:
            return
        if self.reorder and self._corr(self.rng.random()) < self.reorder:
            arrive = now                         # an den Kopf der Schlange, sofort
        else:
            x = self.rng.normal(self.delay, self.jitter) if self.jitter else self.delay
            arrive = max(self.last_send, now + max(x, 0.0))
            self.last_send = arrive
        self.inflight.append((arrive, seq, np.stack([l, r]).astype(np.float32)))

    def step(self, l: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.push(l, r)
        return self.pull()

    def pull(self) -> tuple[np.ndarray, np.ndarray]:
        """Pakete bis zum Ende dieses Takts einliefern (in Ankunftsreihenfolge), dann einen Slot lesen."""
        t_end = self.t + self.slot
        due = sorted((p for p in self.inflight if p[0] < t_end), key=lambda p: (p[0], p[1]))
        if due:
            self.inflight = [p for p in self.inflight if p[0] >= t_end]
        for _, seq, data in due:
            if self.last_seq is None:
                lost = 0
            else:
                lost = seq - self.last_seq - 1
                if lost < 0:
                    self.out_of_order += 1
                    self._ooo_run += 1
                    if self._ooo_run > 5:           # mehr als 5 in Folge: Anfangszustand, nächstes Paket gilt
                        self.last_seq = None
                    continue
            self._ooo_run = 0
            self.lost += lost
            self.last_seq = seq
            self.jb.insert(data, lost * self.slot)
        out = self.jb.read()
        self.t = t_end
        return out[0], out[1]
