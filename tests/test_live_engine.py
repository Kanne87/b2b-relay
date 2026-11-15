"""Live-Engine + JACK-Anbindung gegen den FakeJack-Prüfstand (ohne jackd)."""

import numpy as np
import pytest

from b2b_relay.live.config import LiveConfig
from b2b_relay.sim.synth import Track
from b2b_relay.timeline import frac_read

from live_rig import FS, PERIOD, Rig

LAG = int(0.1 * FS)


@pytest.fixture(scope="module")
def stereo():
    m = Track(seed=7, bars=60).mix.astype(np.float32)
    return m, (0.7 * np.roll(m, 12345)).astype(np.float32)


def make_gen(stereo, lag_b=LAG):
    L, R = stereo

    def seg(x, s, n, lag=0):
        i = np.arange(s, s + n) - lag
        return np.where(i >= 0, x[np.clip(i, 0, len(x) - 1)], 0.0).astype(np.float32)

    def gen(s, n):
        return {"A": (seg(L, s, n), seg(R, s, n)), "B": (seg(L, s, n, lag_b), seg(R, s, n, lag_b))}
    return gen


def xcorr_delay(ref, out, lo, hi):
    """Verzögerung (Samples) von ``out`` gegen ``ref`` im Bereich [lo, hi], per Korrelation über ein Fenster."""
    seg = out[20 * FS:24 * FS]
    best, arg = -1.0, 0
    for d in range(lo, hi, 1):
        r = ref[20 * FS - d:24 * FS - d]
        c = float(np.dot(seg, r))
        if c > best:
            best, arg = c, d
    return arg


def test_discovery_roles_and_return_path_routing(stereo):
    rig = Rig(("A", "B", "TV"), cfg=LiveConfig(router_interval_s=0.0, roles={"TV": "listener"}))
    rig.run(1.0, make_gen(stereo))
    names = {c["name"]: c for c in rig.io.clients()}
    assert names["A"]["role"] == names["B"]["role"] == "dj" and names["TV"]["role"] == "listener"
    assert rig.engine.snapshot().get("live") in (None, "A")
    rig.run(1.0, make_gen(stereo))
    assert rig.engine.snapshot()["live"] == "A"
    w = rig.world
    # DJ → unsere Eingänge; B hört den Roh-Feed von A direkt (JACK-Verbindung), Hörer das Programm
    assert w.sources_of("b2b-relay:in_A_L") == {"A:receive_1"}
    assert w.sources_of("b2b-relay:in_B_R") == {"B:receive_2"}
    assert w.sources_of("B:send_1") == {"A:receive_1"} and w.sources_of("B:send_2") == {"A:receive_2"}
    assert w.sources_of("A:send_1") == set()                       # der Live-DJ hört nichts zurück
    assert w.sources_of("TV:send_1") == {"b2b-relay:program_L"}
    assert w.sources_of("b2b-relay:in_TV_L") == set()              # Hörer liefern kein Programm
    assert {c["name"]: c["hears"] for c in rig.io.clients()} == {"A": None, "B": "feed:A", "TV": "program"}


def test_program_is_delayed_input_and_handover_switches_return_path(stereo):
    rig = Rig(("A", "B"))
    gen = make_gen(stereo)
    rig.run(20.0, gen)
    st = rig.engine.snapshot()
    assert st["live"] == "A" and st["sources"]["B"]["locked"]
    assert st["sources"]["B"]["rtt_ms"] == pytest.approx(100.0, abs=1.0)
    res = rig.engine.handover("B")
    assert res["accepted"], res
    rig.run(20.0, gen)
    st = rig.engine.snapshot()
    assert st["live"] == "B" and st["previous"] == "A"
    w = rig.world
    # nach der Übergabe hört A den Feed von B, B nichts
    assert w.sources_of("A:send_1") == {"B:receive_1"}
    assert w.sources_of("B:send_1") == set()
    # Programmausgang: Verzögerung ~ Zielpuffer (3 s) + Anlauf (2 Blöcke Vorrat + Block), beide Kanäle
    prog = rig.program()
    L, R = stereo
    d = xcorr_delay(L, prog[0], 3 * FS, 3 * FS + 12000)
    assert 3.0 * FS <= d <= 3.0 * FS + 12000
    ref_l = L[20 * FS - d:24 * FS - d]
    ref_r = R[20 * FS - d:24 * FS - d]
    assert np.max(np.abs(prog[0, 20 * FS:24 * FS] - ref_l)) < 5e-3
    assert np.max(np.abs(prog[1, 20 * FS:24 * FS] - ref_r)) < 5e-3
    # Drift bleibt klein, Gutschrift der Übergabe stimmt (Lag-Zuwachs ≈ RTT)
    dr = st["drift"]
    assert dr["anchored"] and abs(dr["drift_ms"]) < 2.0
    assert dr["credits_ms"] == pytest.approx(100.0, abs=2.0)
    assert st["jack"]["xruns"] == 0 and st["jack"]["cb_errors"] == 0 and st["underruns"] == 0
    assert st["worker"]["in_dropped"] == {"A": 0, "B": 0} and st["worker"]["out_dropped"] == 0
    assert rig.engine.rt.out_underruns == 0


def test_drift_is_measured_and_regulated_against_wall_clock(stereo):
    """JACK-Uhr läuft 300 ppm schneller als CLOCK_MONOTONIC: ohne Eingriff wüchsen 18 ms/min."""
    rig = Rig(("A", "B"), clock_ppm=300.0)
    rig.run(70.0, make_gen(stereo))
    dr = rig.engine.snapshot()["drift"]
    assert abs(dr["raw_drift_ms"]) == pytest.approx(21.0, rel=0.25)
    assert abs(dr["drift_ms"]) < 2.0 and dr["max_abs_ms"] < 3.0
    assert abs(dr["corrected_ms"]) == pytest.approx(abs(dr["raw_drift_ms"]), rel=0.2)
    assert dr["ppm_60s"] == pytest.approx(300.0, rel=0.25) or dr["ppm_60s"] == pytest.approx(-300.0, rel=0.25)


def test_xrun_callback_and_callback_errors_are_counted(stereo):
    rig = Rig(("A",))
    rig.run(0.5, make_gen(stereo))
    rig.io._xrun(120.0)
    rig.io._xrun(80.0)
    assert rig.engine.rt.xruns == 2 and rig.engine.rt.cb_errors == 0
    rig.world.ports["b2b-relay:in_A_L"].buf = None      # kaputter Port: der Callback darf nicht werfen
    rig.io._process(PERIOD)
    assert rig.engine.rt.cb_errors == 1
    rig.io._out = None                                  # Ausgang weg: ebenfalls gezählt
    rig.io._process(PERIOD)
    assert rig.engine.rt.cb_errors == 2


def test_roles_change_and_live_dj_cannot_become_listener(stereo):
    rig = Rig(("A", "B"))
    rig.run(2.0, make_gen(stereo))
    with pytest.raises(ValueError):
        rig.io.set_role("A", "listener")
    info = rig.io.set_role("B", "listener")
    assert info.role == "listener"
    assert rig.world.sources_of("B:send_1") == {"b2b-relay:program_L"}
    assert rig.world.sources_of("b2b-relay:in_B_L") == set()
    rig.io.set_role("B", "dj")
    assert rig.world.sources_of("B:send_1") == {"A:receive_1"}
    with pytest.raises(KeyError):
        rig.io.set_role("X", "dj")


def test_client_leaves_and_returns(stereo):
    rig = Rig(("A", "B"))
    gen = make_gen(stereo)
    rig.run(3.0, gen)
    rig.world.remove_remote("B")
    rig.run(1.0, lambda s, n: {"A": gen(s, n)["A"]})
    assert [c for c in rig.io.clients() if c["name"] == "B"][0]["present"] is False
    rig.world.add_remote("B")
    rig.run(1.0, gen)
    assert [c for c in rig.io.clients() if c["name"] == "B"][0]["present"] is True
    assert rig.world.sources_of("B:send_1") == {"A:receive_1"}


def test_recording_writes_program_wav(stereo, tmp_path):
    from b2b_relay.live.wav import read_wav
    cfg = LiveConfig(router_interval_s=0.0, record_dir=str(tmp_path), record_format="float32",
                     record_only_live=False, record_segment_s=1e6)
    rig = Rig(("A", "B"), cfg=cfg)
    rig.engine._open_record()
    rig.run(8.0, make_gen(stereo))
    rig.engine.stop()
    wavs = list(tmp_path.glob("program-*.wav"))
    assert len(wavs) == 1
    fs, a = read_wav(wavs[0])
    assert fs == FS and a.shape[1] == 2 and abs(len(a) - rig.engine.proc) < 10
    assert np.abs(a[int(4 * FS):]).max() > 0.05
    import json
    lines = [json.loads(x) for x in wavs[0].with_suffix(".trace.jsonl").read_text().splitlines()]
    blk = [x for x in lines if x["k"] == "blk"]
    assert len(blk) == rig.engine.chunks and blk[1]["i"] == rig.cfg.chunk
    assert all(abs(b["Tc"] - blk[0]["Tc"] - b["i"]) < 4 for b in blk[:50])    # Rate 1: Tc läuft mit dem Mitschnitt


def test_worker_thread_path(stereo):
    """Echter Worker-Thread: Befehle laufen über die Schlange, das Programm kommt am Ausgang an."""
    import time
    rig = Rig(("A", "B"))
    gen = make_gen(stereo)
    rig.engine.start()
    try:
        for _ in range(int(6 * FS / PERIOD)):
            for name, (l, r) in gen(rig.cyc * PERIOD, PERIOD).items():
                rig.feed(name, l, r)
            rig.cycle()
            time.sleep(0.0005)
        time.sleep(0.3)
        assert "accepted" in rig.engine.handover("B")
        snap = rig.engine.snapshot()
        assert snap["started"] and snap["worker"]["error"] is None
        assert rig.engine.chunks > 50
    finally:
        rig.engine.stop()
    assert np.abs(rig.program()[:, 4 * FS:]).max() > 0.05


# ---------------------------------------------------------------- Aufnahme: Segmente, Fehler, Platz (S3904 S0)

def _rec(tmp_path, **kw):
    from b2b_relay.live.recorder import Recorder
    ev = []
    cfg = LiveConfig(record_dir=str(tmp_path), record_only_live=False, **kw)
    return Recorder(cfg, 1000, lambda kind, **d: ev.append({"kind": kind, **d})), ev


def test_failing_writer_does_not_stop_engine(stereo, tmp_path):
    """Nachbildung der 4-GB-Grenze: Schreiber bricht nach N Bytes mit struct.error ab. Vor dem Fix blieb der
    Worker stehen (Block wurde endlos wiederholt, chunks/proc standen)."""
    import struct
    from b2b_relay.live.wav import WavWriter

    class Limited(WavWriter):
        def write(self, chans):
            super().write(chans)
            if self.frames * 4 > 200_000:
                raise struct.error("'I' format requires 0 <= number <= 4294967295")

    cfg = LiveConfig(router_interval_s=0.0, record_dir=str(tmp_path), record_only_live=False)
    rig = Rig(("A", "B"), cfg=cfg)
    rig.engine.rec._factory = Limited
    rig.engine._open_record()
    rig.run(15.0, make_gen(stereo))
    eng = rig.engine
    assert eng.chunks > 100 and eng.proc > 10 * FS        # Worker zählt weiter
    assert eng.last_error is None
    snap = eng.snapshot()
    errs = [e for e in snap["events"] if e["kind"] == "recording_error"]
    assert errs and "4294967295" in errs[0]["reason"]
    assert snap["recording_info"]["errors"] >= 1
    assert len(list(tmp_path.glob("program-*.wav"))) >= 2   # nach dem Fehler neues Segment


def test_segments_are_sample_exact(tmp_path):
    from b2b_relay.live.wav import read_wav
    rec, _ = _rec(tmp_path, record_segment_s=2.5)
    rng = np.random.default_rng(1)
    sig = rng.uniform(-0.9, 0.9, (2, 1000 * 12)).astype(np.float32)
    k = 250
    for i in range(0, sig.shape[1], k):
        rec.write(sig[0, i:i + k], sig[1, i:i + k], k, True, lambda idx: ['{"k":"blk","i":%d}' % idx])
    rec.close()
    wavs = sorted(tmp_path.glob("program-*.wav"))
    assert len(wavs) == 5 and [w.name for w in wavs] == sorted(w.name for w in wavs)
    parts = [read_wav(w)[1] for w in wavs]
    got = np.concatenate(parts)
    want = np.clip(np.round(sig.T * 32768), -32768, 32767) / 32768
    assert got.shape == want.shape and np.array_equal(got, want.astype(np.float32))
    for w in wavs:                                           # Trace wechselt mit, Index je Segment ab 0
        first = w.with_suffix(".trace.jsonl").read_text().splitlines()[0]
        assert '"i":0' in first.replace(" ", "")


def test_size_limit_deletes_oldest_own_files_only(tmp_path):
    rec, _ = _rec(tmp_path, record_segment_s=1.0, record_max_mb=1)
    (tmp_path / "notes.txt").write_text("fremd")
    (tmp_path / "program-old.txt").write_text("fremd, anderes Muster")
    (tmp_path / "other.wav").write_bytes(b"x" * 3_000_000)
    sig = np.zeros(1000, np.float32)
    for _ in range(40):                                      # 40 Segmente à 4 kB Nutzdaten ... zu klein: Platz füllen
        rec.write(sig, sig, 1000, True, lambda i: ["{}"])
    rec.close()
    own = sorted(tmp_path.glob("program-*.wav"))
    assert len(own) == 40
    # jetzt Obergrenze unterschreiten: großes altes Segment anlegen
    big = tmp_path / "program-00000000-000000-0001.wav"
    big.write_bytes(b"\0" * 900_000)
    big2 = tmp_path / "program-00000000-000001-0001.wav"
    big2.write_bytes(b"\0" * 900_000)
    rec2, _ = _rec(tmp_path, record_segment_s=1.0, record_max_mb=1)
    rec2.write(sig, sig, 1000, True, None)
    rec2.close()
    assert not big.exists() and not big2.exists()            # älteste weg
    assert (tmp_path / "notes.txt").exists() and (tmp_path / "program-old.txt").exists()
    assert (tmp_path / "other.wav").exists()
    n, total = rec2.stats()
    assert total <= 1024 * 1024 and n >= 1


def test_only_live_records_in_new_segment(tmp_path):
    from b2b_relay.live.recorder import Recorder
    cfg = LiveConfig(record_dir=str(tmp_path), record_only_live=True)
    rec = Recorder(cfg, 1000, lambda *a, **k: None)
    z = np.zeros(1000, np.float32)
    for live in [False] * 3 + [True] * 3 + [False] * 6 + [True] * 2:
        rec.write(z, z, 1000, live, None)
    rec.close()
    assert len(list(tmp_path.glob("program-*.wav"))) == 2    # nichts vor der Quelle, neues Segment nach der Pause
