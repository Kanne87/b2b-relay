"""Regression S3904 (Live-Probe 13.11.2026): Ein DJ, der sich erst nach dem Start der Engine verbindet, ergab ein
stummes Programm. Der Quellring des Mixers begann bei Index 0 statt bei der Wanduhr der Anmeldung; die Leseposition
(Wanduhr − Puffer) lag dauerhaft hinter ``ring.end``, ``underruns`` stieg je Block, ``buffer_s`` meldete trotzdem 3,0.

Nachgestellt wie im Befund: Live-Engine, eine einzelne Quelle ohne Remote Name (``__ffff_<ip>``), Verbindung 13 s
nach dem Start, Rolle dj, automatisch live. Abnahme am Programm-Audio (nicht stumm, Korrelation mit der Quelle).
"""

import numpy as np
import pytest

from b2b_relay.live.config import LiveConfig
from b2b_relay.mixer import RelayConfig, RelayMixer
from b2b_relay.sim.synth import Track

from live_rig import FS, PERIOD, Rig

NONAME = "__ffff_10.0.0.1"          # so heißt ein JackTrip-Client ohne Remote Name im Hub ('__ffff_<ip>')


@pytest.fixture(scope="module")
def music():
    return (0.3 * Track(seed=7, bars=60).mix).astype(np.float32)


def best_delay(ref: np.ndarray, out: np.ndarray, end: int, lo: int, hi: int, win: int = 2 * FS):
    """Verzögerung von ``out`` gegen ``ref`` in [lo, hi] für das Fenster ``out[end-win:end]`` und die normierte
    Korrelation dort (1 = identisch)."""
    seg = out[end - win:end].astype(np.float64)
    best, arg = -2.0, lo
    for d in range(lo, hi):
        r = ref[end - win - d:end - d].astype(np.float64)
        den = float(np.linalg.norm(seg) * np.linalg.norm(r))
        c = float(seg @ r) / den if den > 0 else 0.0
        if c > best:
            best, arg = c, d
    return arg, best


def test_mixer_source_registered_late_is_read_at_wall_clock(music):
    """Kern ohne Engine: Quelle wird nach 13 s Laufzeit angemeldet und aktiv; das Programm ist die Quelle,
    um den Zielpuffer verzögert."""
    blk = 2048
    m = RelayMixer(RelayConfig(sample_rate=FS, buffer_s=3.0, target_buffer_s=3.0, block_size=blk,
                               search_max_s=3.0))
    w = 0
    while w < 13 * FS:
        m.process({}, blk)
        w += blk
    m.register_source("X")
    m.set_active("X")
    m.retime(-10 * FS)                     # Vorlauf wie in der Engine (B2B_DRIFT_BASE_S)
    w0 = w
    outs = []
    for _ in range(int(9 * FS / blk)):
        i = np.arange(w - w0, w - w0 + blk) % len(music)
        outs.append(m.process({"X": music[i]}, blk))
        w += blk
    out = np.concatenate(outs)
    u0 = m.status.underruns
    m.process({"X": music[:blk]}, blk)
    assert m.status.underruns == u0                       # eingeschwungen: kein Unterlauf mehr
    d, c = best_delay(music, out, len(out), 3 * FS - 200, 3 * FS + 200)
    assert d == 3 * FS and c > 0.999
    assert m.supply_s == pytest.approx(m.buffer_s, abs=1e-3)


@pytest.mark.parametrize("name", [NONAME, "A"])
def test_live_dj_joining_after_start_is_heard_in_program(music, name):
    rig = Rig((), cfg=LiveConfig(router_interval_s=0.0))

    def gen(s, n):
        i = np.arange(s, s + n) % len(music)
        x = music[i]
        return {name: (x, (0.8 * x).astype(np.float32))}

    rig.run(13.0, gen)                                    # Dienst läuft, noch kein Client
    rig.world.add_remote(name)
    rig.run(8.0, gen)
    st = rig.engine.snapshot()
    assert st["live"] == name and not st["no_source"]
    assert [c["role"] for c in rig.io.clients()] == ["dj"]
    u0 = st["underruns"]
    rig.run(3.0, gen)
    st = rig.engine.snapshot()
    prog = rig.program()
    end = prog.shape[1]
    # Programm führt Ton: Pegel, Korrelation mit der Quelle (Verzögerung ≈ Zielpuffer + Anlauf)
    assert np.abs(prog[:, -3 * FS:]).max() > 0.05
    ref = music[np.arange(end) % len(music)]
    d, c = best_delay(ref, prog[0], end, 3 * FS, 3 * FS + 12000)
    assert c > 0.99, (d, c)
    assert st["underruns"] == u0                          # eingeschwungen: kein Unterlauf mehr
    assert st["program_level_db"] > -40.0 and st["program_silent_s"] == 0.0
    b = st["buffer"]
    assert b["supply_s"] == pytest.approx(b["buffer_s"], abs=0.01)
    assert b["buffer_s"] == pytest.approx(3.0, abs=0.1)


def test_second_dj_joining_late_gets_rtt_and_takes_over(music):
    """Auch die Schätzung braucht den Ring auf der Wanduhr: B verbindet sich 13 s nach A und hört A mit 100 ms."""
    lag = int(0.1 * FS)
    rig = Rig(("A",), cfg=LiveConfig(router_interval_s=0.0))

    def gen(s, n):
        i = np.arange(s, s + n)
        a = music[i % len(music)]
        b = np.where(i - lag >= 0, music[(i - lag) % len(music)], 0.0).astype(np.float32)
        return {"A": (a, a), "B": (b, b)}

    rig.run(13.0, gen)
    rig.world.add_remote("B")
    rig.run(16.0, gen)
    st = rig.engine.snapshot()
    assert st["sources"]["B"]["locked"], st["sources"]
    assert st["sources"]["B"]["rtt_ms"] == pytest.approx(100.0, abs=1.0)
    assert rig.engine.handover("B")["accepted"]
    rig.run(8.0, gen)
    st = rig.engine.snapshot()
    assert st["live"] == "B" and st["program_level_db"] > -40.0
    prog = rig.program()
    assert np.abs(prog[:, -2 * FS:]).max() > 0.05


def test_program_level_reports_silence_without_source():
    rig = Rig((), cfg=LiveConfig(router_interval_s=0.0))
    rig.run(2.0, lambda s, n: {})
    st = rig.engine.snapshot()
    assert st["no_source"] and st["program_level_db"] <= -100.0 and st["program_silent_s"] >= 1.5
    assert st["buffer"]["supply_s"] is None
