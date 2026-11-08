"""Reproduktion der Phasensprünge aus dem Integrationstest (Run 42515682729) ohne Docker.

Drei Test-DJs (``DjCore`` aus ``tests/integration/client.py``) mit unterschiedlicher RTT hängen am
Live-Server (FakeJack-Prüfstand); der Router schaltet den Rückkanal wie im Betrieb. Vor dem Fix sprang die
Phase bei B→C um ≈ RTT(B) und bei C→A um ≈ RTT(A)+RTT(C) (die Lag-Schätzung der wartenden Quellen
folgte dem Rückkanal-Wechsel nicht, die erste Quelle wurde nach der Abgabe nie wieder gemessen), und der
Testclient setzte jeden Kick im Mittel einen halben Block zu früh (Vorecho des Ausklangs).
"""

import numpy as np
import pytest

import itest_sim
import signals as sg
from client import DjCore

FS = 48000
RTT = {"A": 20.0, "B": 35.0, "C": 50.0}
CYCLE_MS = 256 / FS * 1e3           # ein Takt des Prüfstands (Weitergabe DJ → Server) kommt zur RTT dazu


def test_testclient_places_own_onsets_on_the_heard_grid():
    heard = sg.render_track("A", 40, 1000.0)
    dj = DjCore("B")
    outs = []
    for k in range(0, len(heard) - 256, 256):
        blk = heard[k:k + 256]
        outs.append(dj.process(blk, blk)[0].copy())
        if (k // 256) % 37 == 0:
            dj.control("A", "B", k / FS)
    own = np.concatenate(outs) - heard[:len(outs) * 256]       # g_feed = 1: Rest sind die eigenen Kicks
    ph_heard, _ = sg.grid_phase(sg.detect_onsets(heard[20 * FS:38 * FS]) + 20 * FS)
    ph_own, r = sg.grid_phase(sg.detect_onsets(own[20 * FS:38 * FS]) + 20 * FS)
    assert r > 0.999
    assert abs(sg.wrap(ph_own - ph_heard)) / FS * 1e3 < 0.1     # vorher −2,5 ms


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    return itest_sim.run_session(tmp_path_factory.mktemp("itest"), RTT, handovers=3)


def test_three_djs_with_different_rtt_hand_over_without_phase_jump(session):
    per = session["per_handover"]
    assert [(p["from"], p["to"]) for p in per] == [("A", "B"), ("B", "C"), ("C", "A")]
    jumps = [p["phase_jump_ms"] for p in per]
    assert all(j is not None and abs(j) <= 1.0 for j in jumps), jumps       # vorher −2 / +8 / +117 ms
    assert session["handovers_refused"] == 0
    assert session["acceptance"]["passed"], session["acceptance"]


def test_lag_estimates_follow_the_return_path(session):
    """Lag des Ziels = Lag der Quelle, die es hört, + eigene RTT (der Lag summiert sich)."""
    lags = session["lags_ms"]
    true, prev = [], 0.0
    for p in session["per_handover"]:
        prev += RTT[p["to"]] + CYCLE_MS
        true.append(prev)
    assert lags == pytest.approx(true, abs=0.5), (lags, true)
