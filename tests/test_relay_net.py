"""Regression S3904: Übergaben unter Netzstörung ohne Docker.

Nachgebildet ist die Upload-Strecke des Integrationstests (``tests/netsim.py``: ``tc netem … rate 1000mbit`` +
JackTrip-Empfang am Hub mit Jitterpuffer-Strategie 1) mit den Profilen aus ``tests/integration/run.py``; der
Rest wie in ``tests/itest_sim.py`` (Live-Server auf dem FakeJack-Prüfstand, drei Test-DJs aus ``client.py``).

Fehlerbild in CI-Run 13 (42524877312) und in dieser Nachbildung vor dem Fix (main 12e1e8c):

* Eine wartende Quelle rastet auf einen falschen Peak ein (Run 13 wan40: B bei 266,7 ms statt ≈ 70 ms;
  Nachbildung wan120: C bei 240,7 ms statt ≈ 181 ms, A bei 1973 ms). Die Nachführung sucht nur ±3 ms um die
  Vorhersage und findet nie zurück; die Driftrate läuft an die Grenze 500 ppm. Folge: Phasensprung = −Fehler mod
  Beat (Run 13: +94,8 ms; Nachbildung −89,6 / −65,1 ms).
* Die falsche Driftrate geht über jede Übergabe an alle Quellen über (Run 13 wan40: 442–500 ppm, clean −15,5 ppm)
  und erscheint als Sprung ≈ −Rate · 30 s (Abstand der Messfenster): −10…−14 ms bzw. +0,47 ms.
* JackTrip gibt für verspätete und verlorene Pakete Nullen aus: Lücken von 5,3 ms und Vielfachen im Programm.

Vor dem Fix in genau diesen Sitzungen (main 12e1e8c mit diesem Testaufbau): wan40 Sprünge 18,1 / 39,8 / 15,5 ms;
wan120 6,2 / −7,3 / −65,1 ms und 382 Lücken. JackTrip-Queue des Hubs fest 12 Slots wie im Integrationstest.
"""

import numpy as np
import pytest

import itest_sim
from netsim import NetemUplink

pytestmark = pytest.mark.slow

PROFILES = {   # netem-Parameter wie tests/integration/run.py
    "wan40": dict(delay_ms=40, jitter_ms=5),
    "wan120": dict(delay_ms=120, jitter_ms=20, loss=0.005, reorder=0.05, reorder_corr=0.5),
}


def _session(tmp_path, profile: str, qlen: int, seed: int, handovers: int = 3) -> dict:
    links: dict[str, NetemUplink] = {}
    status: list[dict] = []

    def uplink(name):
        links[name] = NetemUplink(seed=seed * 10 + ord(name) - 64, qlen=qlen, **PROFILES[profile])
        return links[name]

    res = itest_sim.run_session(tmp_path, {"A": 10.67, "B": 10.67, "C": 10.67}, handovers=handovers,
                                warmup_s=45, lead_s=30, hold_s=60, settle_s=20, uplink=uplink, profile=profile,
                                on_status=lambda t, s: status.append(s) if s.get("started") else None)
    res["status"] = status
    res["links"] = links
    return res


def _rates(res) -> set:
    return {v["drift_ppm"] for s in res["status"] for v in s["sources"].values() if v["drift_ppm"] is not None}


def test_wan40_fixed_queue_hands_over_without_jump(tmp_path):
    """Jitter 5 ms, JackTrip-Queue fest 12 Slots (Empfehlung für den Hub): die Strecke verschiebt nichts mehr,
    jeder Sprung wäre Fehler des Relays. Vorher 18 / 40 / 15 ms (falsche Einrastung, Driftrate)."""
    res = _session(tmp_path, "wan40", 12, seed=1)
    jumps = [p["phase_jump_ms"] for p in res["per_handover"]]
    assert len(jumps) == 3 and all(j is not None and abs(j) < 1.0 for j in jumps), jumps
    assert res["handovers_refused"] == 0
    assert res["summary"]["gaps_total"] == 0
    assert _rates(res) == {0.0}, _rates(res)            # keine Driftrate hinter dem Hub


def test_wan120_hands_over_without_false_lock_and_without_gaps(tmp_path):
    """Verlust 0,5 %, Umordnung 5 %, Queue 12: keine falsche Einrastung (der Lag wächst je Übergabe um die RTT,
    ≈ 175–195 ms), keine Lücke im Programm (Aussetzer verschleiert), Sprung unter 10 ms. Der Rest kommt von
    der Strecke: JackTrip verschiebt den Feed bei Verlust und Umordnung um ganze Slots (Nullmessung ohne
    Übergabe bis 2,4 ms, mit Queue auto bis 6,7 ms). Vorher 6,2 / −7,3 / −65,1 ms (Einrastung bei 1973 ms) und
    382 Lücken."""
    res = _session(tmp_path, "wan120", 12, seed=1)
    jumps = [p["phase_jump_ms"] for p in res["per_handover"]]
    assert len(jumps) == 3 and all(j is not None and abs(j) < 10.0 for j in jumps), jumps
    steps = np.diff([0.0] + res["lags_ms"])
    assert all(150.0 < s < 215.0 for s in steps), res["lags_ms"]
    assert res["summary"]["gaps_total"] == 0
    assert res["summary"]["concealed"] > 50                # die Strecke hat ausgesetzt, das Programm nicht
    assert sum(lk.lost for lk in res["links"].values()) > 100
    assert _rates(res) == {0.0}, _rates(res)
