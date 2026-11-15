"""Regression S4006 (Live-Server 15.11.2026): Ein DJ trennt sich, das Relay führt ihn weiter als verbundene und
live spielende Quelle (locked, stale_s null, live=<name>), die Aufnahme schreibt Stille, 'retime' läuft weiter.

Nachgestellt mit FakeJack: Ports des Clients verschwinden (``remove_remote``). Nach der Karenz
(``B2B_SOURCE_GRACE_S``) gilt die Quelle als getrennt."""

import numpy as np
import pytest

from b2b_relay.live.config import LiveConfig
from b2b_relay.sim.synth import Track

from live_rig import FS, Rig

LAG = int(0.1 * FS)


@pytest.fixture(scope="module")
def music():
    return (0.3 * Track(seed=7, bars=60).mix).astype(np.float32)


def make_gen(music):
    def gen(s, n):
        i = np.arange(s, s + n)
        a = music[i % len(music)]
        b = np.where(i - LAG >= 0, music[(i - LAG) % len(music)], 0.0).astype(np.float32)
        return {"A": (a, a), "B": (b, b)}
    return gen


def retimes(rig):
    return sum(1 for e in rig.engine.master.events if e.kind == "retime")


def kinds(rig, kind):
    return [e for e in rig.engine.snapshot()["events"] if e["kind"] == kind]


@pytest.fixture(scope="module")
def lost(music, tmp_path_factory):
    """Zwei DJs, A live, B wartet eingerastet; A trennt sich; 12 s später (Karenz 5 s) ist der Stand erreicht."""
    rec = tmp_path_factory.mktemp("rec")
    cfg = LiveConfig(router_interval_s=0.0, record_dir=str(rec), source_grace_s=5.0)
    rig = Rig(("A", "B"), cfg=cfg)
    gen = make_gen(music)
    rig.run(20.0, gen)
    st = rig.engine.snapshot()
    assert st["live"] == "A" and st["sources"]["B"]["locked"]
    assert st["recording_info"]["state"] == "recording"
    rig.world.remove_remote("A")
    rig.run(3.0, gen)                                   # innerhalb der Karenz: noch live
    during = rig.engine.snapshot()
    rig.run(9.0, gen)
    return rig, during, gen


def test_within_grace_source_is_still_live(lost):
    _, during, _ = lost
    assert during["live"] == "A" and "A" in during["sources"]


def test_after_grace_no_source_and_event(lost):
    rig, _, _ = lost
    st = rig.engine.snapshot()
    assert st["live"] is None and st["no_source"] is True
    assert "A" not in st["sources"] and "B" in st["sources"]
    ev = kinds(rig, "source_lost")
    assert len(ev) == 1 and ev[0]["source"] == "A" and ev[0]["was_live"] is True
    assert [c for c in rig.io.clients() if c["name"] == "A"][0]["present"] is False


def test_program_is_silent_and_no_retime_flood(lost):
    rig, _, gen = lost
    st = rig.engine.snapshot()
    assert st["program_level_db"] <= -100.0 and st["program_silent_s"] >= 3.0
    n0 = retimes(rig)
    rig.run(5.0, gen)
    assert retimes(rig) == n0                           # ohne Quelle keine retime-Ereignisse


def test_recording_waits_after_loss(lost):
    rig, _, _ = lost
    assert rig.engine.snapshot()["recording_info"]["state"] == "waiting"


def test_handover_to_waiting_dj_after_loss(lost):
    rig, _, gen = lost
    r = rig.engine.handover("B")
    assert r["accepted"], r
    rig.run(8.0, gen)
    st = rig.engine.snapshot()
    assert st["live"] == "B" and not st["no_source"]
    assert st["program_level_db"] > -40.0


def test_reconnect_same_name_is_new_source(music):
    rig = Rig(("A",), cfg=LiveConfig(router_interval_s=0.0, source_grace_s=2.0))
    gen = make_gen(music)
    rig.run(12.0, gen)
    assert rig.engine.snapshot()["live"] == "A"
    rig.world.remove_remote("A")
    rig.run(5.0, gen)
    st = rig.engine.snapshot()
    assert st["no_source"] and "A" not in st["sources"]
    rig.world.add_remote("A")
    rig.run(10.0, gen)
    st = rig.engine.snapshot()
    # niemand live → wie der erste DJ nach dem Start: automatisch live, sauber und ohne Altzustand
    assert st["live"] == "A" and not st["no_source"]
    assert st["sources"]["A"]["stale_s"] is None or st["sources"]["A"]["stale_s"] < 10
    u0 = st["underruns"]
    rig.run(4.0, gen)
    st = rig.engine.snapshot()
    assert st["program_level_db"] > -40.0 and st["underruns"] == u0
    assert len(kinds(rig, "source_lost")) == 1


def test_reconnect_within_grace_keeps_source(music):
    rig = Rig(("A",), cfg=LiveConfig(router_interval_s=0.0, source_grace_s=5.0))
    gen = make_gen(music)
    rig.run(10.0, gen)
    rig.world.remove_remote("A")
    rig.run(2.0, gen)
    rig.world.add_remote("A")
    rig.run(8.0, gen)
    st = rig.engine.snapshot()
    assert st["live"] == "A" and not kinds(rig, "source_lost")


def test_grace_from_env():
    assert LiveConfig.from_env({"B2B_SOURCE_GRACE_S": "2.5"}).source_grace_s == 2.5
    assert LiveConfig().source_grace_s == 5.0
    with pytest.raises(ValueError):
        LiveConfig.from_env({"B2B_SOURCE_GRACE_S": "-1"})
