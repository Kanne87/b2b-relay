"""Integrationstest ohne Docker: Live-Server (FakeJack-Prüfstand) + drei Test-DJs aus
``tests/integration/client.py`` (``DjCore``) mit je eigener Netzlaufzeit.

Nachgebildet wird, was im Compose-Stack zwischen Server und Clients liegt: je DJ eine feste Laufzeit
hin (Rückkanal) und zurück (Upload), sonst nichts. Der Router des Servers (``live/jackio.py``) schaltet
den Rückkanal wie im Betrieb: jeder wartende DJ hört den Roh-Feed des live spielenden DJs. Der Ablauf
entspricht ``tests/integration/run.py`` (ctl/next, Vorlauf, POST /handover, Haltezeit), die Auswertung
ist ``tests/integration/analyze.py`` auf dem Mitschnitt des Servers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from b2b_relay.live.config import LiveConfig
from live_rig import FS, PERIOD, Rig

ITEST = Path(__file__).resolve().parent / "integration"
if str(ITEST) not in sys.path:
    sys.path.insert(0, str(ITEST))

import analyze  # noqa: E402
from client import DjCore  # noqa: E402


class Delay:
    """Feste Laufzeit in Samples (Stereo, blockweise)."""

    def __init__(self, samples: int) -> None:
        self.buf = np.zeros((2, samples + PERIOD), np.float32)
        self.d = samples

    def push(self, l: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(l)
        self.buf = np.concatenate([self.buf, np.stack([l, r])], axis=1)
        out = self.buf[:, -self.d - n:len(self.buf[0]) - self.d] if self.d else self.buf[:, -n:]
        self.buf = self.buf[:, -(self.d + PERIOD):]
        return out[0].copy(), out[1].copy()

    step = push


def run_session(out_dir: Path, rtt_ms: dict[str, float], *, handovers: int = 3, warmup_s: float = 20.0,
                lead_s: float = 10.0, hold_s: float = 50.0, settle_s: float = 30.0,
                windows=analyze.WINDOWS, uplink=None, profile: str = "clean",
                cfg: LiveConfig | None = None, on_status=None) -> dict:
    """Eine Sitzung A→B→C→… wie ``run.py``; liefert das Ergebnis von ``analyze.analyze_profile`` plus
    ``lags_ms`` (Lag-Schätzung des Ziels je Übergabe, ms über dem Vorlauf ``drift_base_s``).

    ``uplink(name)`` ersetzt die feste Upload-Laufzeit durch ein Streckenmodell mit ``step(l, r) → (l, r)``
    je Takt (z. B. ``netsim.NetemUplink``: netem + JackTrip-Jitterpuffer); ``rtt_ms`` gilt dann nur für den
    Rückkanal. ``on_status(t_s, snapshot)`` wird alle 2 s mit dem Status des Servers gerufen (wie
    ``status.jsonl`` des Orchestrators)."""
    names = list(rtt_ms)
    cfg = cfg or LiveConfig()
    cfg.router_interval_s, cfg.auto_active, cfg.record_dir = 0.0, False, str(out_dir)
    cfg.record_format, cfg.drift_base_s = "float32", 10.0
    rig = Rig(tuple(names), cfg=cfg)
    rig.engine._open_record()
    djs = {n: DjCore(n) for n in names}
    # RTT je DJ: halb Rückkanal, halb Upload (ganze Samples); dazu ein Takt des Prüfstands
    if uplink is None:
        down = {n: Delay(int(round(rtt_ms[n] * 1e-3 * FS / 2))) for n in names}
        up = {n: Delay(int(round(rtt_ms[n] * 1e-3 * FS)) - int(round(rtt_ms[n] * 1e-3 * FS / 2))) for n in names}
    else:
        down = {n: Delay(int(round(rtt_ms[n] * 1e-3 * FS))) for n in names}
        up = {n: uplink(n) for n in names}
    to_server = {n: (np.zeros(PERIOD, np.float32), np.zeros(PERIOD, np.float32)) for n in names}
    ctl = {"next": ""}
    ctl_every = int(round(0.2 * FS / PERIOD))
    status_every = int(round(2.0 * FS / PERIOD))
    reqs: list[dict] = []

    def step(seconds: float) -> None:
        for _ in range(int(round(seconds * FS / PERIOD))):
            for n in names:
                rig.feed(n, *to_server[n])
            rig.cycle()
            for n in names:
                l = rig.world.ports[f"{n}:send_1"].buf.copy()
                r = rig.world.ports[f"{n}:send_2"].buf.copy()
                lo, ro = djs[n].process(*down[n].push(l, r))
                to_server[n] = up[n].step(lo, ro)
            if on_status is not None and rig.cyc % status_every == 0:
                on_status(rig.cyc * PERIOD / FS, rig.engine.snapshot())
            if rig.cyc % ctl_every == 0:
                live = rig.engine.snapshot().get("live")
                t = rig.cyc * PERIOD / FS
                for d in djs.values():
                    d.control(live, ctl["next"], t)

    step(2.0)
    assert rig.engine.handover(names[0])["accepted"]
    step(warmup_s)
    for k in range(1, handovers + 1):
        frm, to = names[(k - 1) % len(names)], names[k % len(names)]
        ctl["next"] = to
        step(lead_s)
        rec = {"from": frm, "to": to, "attempts": 0, "accepted": False}
        spent = lead_s
        while spent < hold_s - 1.0:
            rec["attempts"] += 1
            res = rig.engine.handover(to)
            if res.get("accepted"):
                rec.update(accepted=True, result=res)
                break
            rec["last_refusal"] = res
            step(1.0)
            spent += 1.0
        reqs.append(rec)
        step(max(hold_s - spent, 0.0))
    step(settle_s)
    rig.engine.stop()
    wav = sorted(out_dir.glob("program-*.wav"))[-1]
    res = analyze.analyze_profile(wav, wav.with_suffix(".trace.jsonl"), profile, None, reqs, windows=windows)
    res["lags_ms"] = [None if p["lag_s"] is None else round((p["lag_s"] - cfg.drift_base_s) * 1e3, 2)
                      for p in res["per_handover"]]
    return res
