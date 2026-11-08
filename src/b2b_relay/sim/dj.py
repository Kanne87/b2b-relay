"""DJ-Modell der Simulation.

Ein DJ erzeugt seinen Master ``M(T)`` auf der Programmzeit-Achse (Labels T):

* eigene Tracks mit Gain-/Bass-Swap-Kurven (vorab gerendert, ``own``),
* beim Übernehmen zusätzlich den Rückkanal ``heard(T)`` = Programm P(T), wie es der Relay-Mixer
  am Live-Rand gerade hat, mit Downlink-Paketverlust, per Tiefpass in Bass und Höhen geteilt
  (EQ) und mit Fader-/Bass-Swap-Kurven ausgeblendet.

Die Laufzeit steckt in ``Stint.lag(w)``: der Feed trifft beim Mixer zur Wanduhr ``w`` ein und
trägt den Programminhalt ``T = w − lag(w)``. ``lag`` = wahrer Lag der Programmquelle zum Zeitpunkt,
als sie P(T) lieferte, + RTT des DJs + Taktdrift seiner Audio-Schnittstelle (ppm) seit dem Anhören.

Der Rückkanal ist der **Roh-Feed** der abgebenden Quelle (``mixer.read_feed``), wie im Betrieb: Er
wird nicht mit den Schätzwerten des Mixers neu abgetastet. Sonst würden Schätzfehler der Driftrate
zur „wahren“ Drift des nächsten DJs und die Drift über viele Übergaben als Irrfahrt wegdriften.
Die Labels T des DJs liegen im Zeitraster der wahren Lags; das Raster des Mixers weicht davon um
den aufsummierten Schätzfehler ab (wenige ms) und ist für die Phasenmessung unerheblich.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mixer import RelayMixer
from ..timeline import frac_read
from .plan import Plan, PHandoff
from .transport import PACKET

try:    # scipy nur für die Simulation
    from scipy.signal import butter, lfilter
except ImportError as e:    # pragma: no cover
    raise ImportError("Die Simulation braucht scipy: pip install -e .[sim]") from e


@dataclass
class Stint:
    """Lag-Funktion eines DJs ab seinem Zuhören (``w_from``)."""

    w_from: float
    ref: tuple[float, float, float] | None   # (l0, rho, t_ref) der Programmquelle, None = Referenz
    rtt: float                                # Samples (Down + Up inkl. Jitter-Puffer)
    eps: float                                # Taktdrift (relativ)
    w_cut: float = float("inf")               # ab hier wirkt die Programmquelle nicht mehr nach

    def arrival(self, t: float) -> float:
        """Wanduhr w, zu der der Programminhalt t im Feed dieses DJs eintrifft (w − lag(w) = t)."""
        w = t + float(self.lag(t + 1.0))
        for _ in range(6):
            w = t + float(self.lag(w))
        return w

    def lag(self, w: np.ndarray | float) -> np.ndarray | float:
        if self.ref is None:
            return np.zeros_like(w, dtype=np.float64) if isinstance(w, np.ndarray) else 0.0
        l0, rho, tref = self.ref
        wl = np.minimum(w - self.rtt, self.w_cut)
        return l0 + rho * (wl - tref) + self.rtt + self.eps * (w - self.w_from)


@dataclass
class _Window:
    ho: PHandoff
    a: int
    b: int
    done: int
    zi: np.ndarray | None = None
    ref: tuple[float, float, float] | None = None   # wahrer Lag der Programmquelle (l0, rho, t_ref)


class SimDJ:
    def __init__(self, name: str, plan: Plan, total_t: int, *, rtt_s: float, eps: float,
                 up_lost: np.ndarray, down_lost: np.ndarray) -> None:
        self.name = name
        self.plan = plan
        self.fs = plan.fs
        self.rtt = rtt_s * self.fs
        self.eps = eps
        self.up_lost = up_lost
        self.down_lost = down_lost
        self.M = np.zeros(total_t, dtype=np.float32)
        self._build_own()
        self.windows = [_Window(ho, *plan.echo_window(ho), plan.echo_window(ho)[0])
                        for ho in plan.handoffs if ho.dst == name]
        self.stints: list[Stint] = []
        if plan.cfg.names[0] == name:
            self.stints.append(Stint(0.0, None, 0.0, 0.0))
        self._b, self._a = butter(2, 200.0, fs=self.fs)
        self.loss_intervals: list[tuple[float, float]] = []   # Programmzeit (Samples)

    # -- Aufbau

    def _build_own(self) -> None:
        plan = self.plan
        for t in plan.tracks:
            if t.dj != self.name:
                continue
            trk = plan.track_of(t)
            i0 = max(0, int(np.floor(t.start)) - 2)
            i1 = int(np.ceil(t.start + trk.n))
            i1 = min(i1, self.M.shape[0])
            T = np.arange(i0, i1, dtype=np.float64)
            gl, gh = plan.own_gains(t, T)
            keep = (gl > 0) | (gh > 0)
            if not keep.any():
                continue
            pos = T - t.start
            seg = gl * frac_read(trk.low, pos) + gh * frac_read(trk.high, pos)
            self.M[i0:i1] += seg.astype(np.float32)

    def start_stint(self, ho: PHandoff, src_stint: Stint) -> Stint:
        """Der DJ beginnt, das Programm zu hören. Der Lag der Programmquelle ist ihr wahrer Lag
        (linear in dem Bereich, in dem sie nichts mehr hört: Anstieg = ihre Taktdrift)."""
        fs = self.fs
        w_s = src_stint.arrival(ho.t_arm)
        l0 = float(src_stint.lag(w_s))
        rho = (float(src_stint.lag(w_s + fs)) - l0) / fs
        tref = w_s
        w_arm = (ho.t_arm + l0 - rho * tref) / (1.0 - rho)
        w_cut = (self.plan.echo_window(ho)[1] + l0 - rho * tref) / (1.0 - rho)
        st = Stint(w_arm, (l0, rho, tref), self.rtt, self.eps, w_cut)
        self.stints.append(st)
        for win in self.windows:
            if win.ho is ho:
                win.ref = (l0, rho, tref)
        return st

    def current_stint(self, w: float) -> Stint | None:
        cur = None
        for st in self.stints:
            if st.w_from <= w:
                cur = st
        return cur

    # -- Rückkanal in den Master einmischen (lazy, in Programmzeit)

    def _ensure(self, t_need: int, mixer: RelayMixer) -> None:
        plan = self.plan
        for win in self.windows:
            if win.done >= win.b or t_need <= win.a:
                continue
            a = win.done
            b = min(t_need, win.b)
            if b <= a:
                continue
            if win.zi is None:
                win.zi = np.zeros(max(len(self._a), len(self._b)) - 1)
            T = np.arange(a, b)
            l0, rho, tref = win.ref
            ws = (T + l0 - rho * tref) / (1.0 - rho)         # Wanduhr-Index im Feed der Quelle
            lo = int(np.floor(ws[0])) - 2
            seg = mixer.read_feed(win.ho.src, lo, int(np.floor(ws[-1])) + 4 - lo)
            heard = frac_read(seg, ws - lo)
            lost = self.down_lost[T // PACKET]
            if lost.any():
                if np.abs(heard[lost]).max(initial=0.0) > 1e-4:
                    for p in np.unique(T[lost] // PACKET):
                        self.loss_intervals.append((float(p * PACKET), float((p + 1) * PACKET)))
                heard[lost] = 0.0
            lp, win.zi = lfilter(self._b, self._a, heard, zi=win.zi)
            hp = heard - lp
            gl, gh = plan.echo_gains(win.ho, T.astype(np.float64))
            self.M[a:b] += (gl * lp + gh * hp).astype(np.float32)
            win.done = b

    # -- Feed für einen Wanduhr-Block

    def feed_block(self, w: int, n: int, mixer: RelayMixer) -> np.ndarray:
        st = self.current_stint(w + n - 1)
        if st is None:
            return np.zeros(n, dtype=np.float32)
        ww = np.arange(w, w + n, dtype=np.float64)
        Tw = ww - st.lag(ww)
        t_hi = int(np.ceil(Tw[-1])) + 4
        if t_hi < 2:
            return np.zeros(n, dtype=np.float32)
        self._ensure(t_hi, mixer)
        i0 = max(int(np.floor(Tw[0])) - 1, 0)
        i1 = min(int(np.floor(Tw[-1])) + 3, self.M.shape[0] - 1)
        if i1 <= i0 + 3:
            return np.zeros(n, dtype=np.float32)
        out = frac_read(self.M[i0:i1 + 1], Tw - i0)
        pk = (ww // PACKET).astype(np.int64)
        lost = self.up_lost[pk]
        if lost.any():
            if np.abs(out[lost]).max(initial=0.0) > 1e-4:
                for p in np.unique(pk[lost]):
                    t_lo = p * PACKET - float(st.lag(float(p * PACKET)))
                    self.loss_intervals.append((t_lo, t_lo + PACKET))
            out[lost] = 0.0
        return out.astype(np.float32)
