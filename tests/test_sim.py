import json
import wave

import numpy as np

from b2b_relay.sim.__main__ import main
from b2b_relay.sim.analysis import find_gaps, kick_phase_errors, max_sample_jump, onset_shift
from b2b_relay.sim.plan import Plan, PlanConfig
from b2b_relay.sim.synth import Track, beat_samples
from b2b_relay.sim.transport import Link
from b2b_relay.timeline import frac_read

FS = 48000


def test_track_is_deterministic_and_seeds_differ():
    a, b, c = Track(1, 8), Track(1, 8), Track(2, 8)
    np.testing.assert_array_equal(a.mix, b.mix)
    assert np.abs(a.mix - c.mix).max() > 0.1
    # Kicks liegen auf dem 174-BPM-Raster (Schritt 1 und 11 jedes Takts)
    bar = 4 * beat_samples()
    first = a.kick_times[:2]
    assert abs(first[0]) < 1e-6 and abs(first[1] - 10 * bar / 16) < 1e-6
    assert not find_gaps(a.mix[:6 * FS], FS)          # nie digitale Stille


def test_link_loss_model_rates_and_validation():
    rng = np.random.default_rng(0)
    link = Link(0.04, 0.01, 0.001)
    mask, st = link.loss_mask(128 * 400_000, rng)
    rate = mask.mean()
    assert 0.0008 < rate < 0.002             # 0,1 % Verlust + wenige verspätete Pakete
    assert st["late"] > 0
    assert Link(0.04, 0.0, 0.0).loss_mask(128 * 1000, rng)[0].sum() == 0
    try:
        Link(0.005, 0.01, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Jitter-Puffer größer als Latenz muss abgelehnt werden")


def test_onset_shift_measures_known_shift():
    t = Track(3, 16)
    ref = lambda pos: frac_read(t.low, pos).astype(float)
    kicks = t.kick_times[6:16]
    for shift_ms in (0.0, 2.5, -3.7):
        prog = frac_read(t.mix, np.arange(len(t.mix), dtype=float) - shift_ms * 48.0)
        res = kick_phase_errors(prog, ref, kicks, FS)
        err = np.array([r["err_ms"] for r in res if r["ok"]])
        assert len(err) >= 8 and np.abs(err - shift_ms).max() < 0.05


def test_gap_and_jump_helpers():
    x = np.ones(10000) * 0.1
    x[3000:3400] = 0.0                      # 8,3 ms
    x[6000:6100] = 0.0                      # 2,1 ms → keine Lücke > 5 ms
    assert find_gaps(x, FS) == [(3000, 400)]
    assert abs(max_sample_jump(x) - 0.1) < 1e-12
    ex = np.zeros(10000, dtype=bool)
    ex[2990:3410] = True
    ex[5990:6110] = True
    assert max_sample_jump(x, ex) == 0.0


def test_plan_is_deterministic_and_alternates():
    p1 = Plan(PlanConfig(handoffs=4, seed=3))
    p2 = Plan(PlanConfig(handoffs=4, seed=3))
    assert [round(t.start, 3) for t in p1.tracks] == [round(t.start, 3) for t in p2.tracks]
    assert [h.dst for h in p1.handoffs] == ["B", "A", "B", "A"]
    assert all(abs(t.human_s) <= 0.005 for t in p1.tracks)
    # je DJ zwei eigene Übergänge vor der Übergabe
    assert [t.kind for t in p1.tracks[:4]] == ["initial", "solo", "solo", "handoff"]


def test_cli_end_to_end(tmp_path, capsys):
    rep_path, wav_path = tmp_path / "r.json", tmp_path / "p.wav"
    rc = main(["--handoffs", "2", "--mix-bars", "4", "--rtt-a-ms", "60", "--rtt-b-ms", "220",
               "--jitter-ms", "12", "--loss", "0.001", "--drift-ppm", "50", "--seed", "2",
               "--target-buffer-s", "0", "--min-gap-s", "0",
               "--out", str(rep_path), "--wav", str(wav_path), "--quiet"])
    rep = json.loads(rep_path.read_text())
    assert rc == 0, rep["summary"]
    assert rep["acceptance"]["passed"]
    assert len(rep["handoffs"]) == 2 and all(h["accepted"] for h in rep["handoffs"])
    for h in rep["handoffs"]:
        assert abs(h["rtt_error_ms"]) < 1.0
        assert abs(h["beat_phase_error_max_abs_ms"]) < 3.0
        assert h["gaps_unexplained"] == 0
        assert "dj_human_error_ms" in h and "new_track_relay_error_ms" in h
    with wave.open(str(wav_path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 48000)
        assert w.getnframes() > 40 * 48000


def test_cli_elastic_output_end_to_end(tmp_path):
    """Mit Regelung: Puffer/Rate/Zuschauer-Verzögerung im Report, Messungen in Programmzeit."""
    rep_path = tmp_path / "r.json"
    rc = main(["--handoffs", "2", "--mix-bars", "4", "--rtt-a-ms", "40", "--rtt-b-ms", "250",
               "--jitter-ms", "12", "--loss", "0.001", "--drift-ppm", "50", "--seed", "2",
               "--buffer-s", "2.9", "--target-buffer-s", "3", "--max-dev", "0.005",
               "--min-gap-s", "0", "--settle-s", "60", "--out", str(rep_path), "--quiet"])
    rep = json.loads(rep_path.read_text())
    assert rc == 0, (rep["summary"], rep["acceptance"])
    acc = rep["acceptance"]
    for key in ("buffer_never_below_min", "viewer_delay_below_4s", "rate_within_max_dev",
                "buffer_ends_near_target", "no_unexplained_gap_over_5ms", "beat_phase_below_3ms"):
        assert acc[key], key
    assert rep["buffer"]["min_s"] >= 1.0 and rep["rate"]["max_abs_dev"] <= 0.005 + 1e-9
    assert rep["viewer_delay"]["max_s"] < 4.0 and rep["series"]
    for h in rep["handoffs"]:
        assert h["accepted"] and h["buffer_after_expected_s"] >= 1.0
        assert abs(h["beat_phase_error_max_abs_ms"]) < 3.0 and h["gaps_unexplained"] == 0


def test_plan_min_gap_stretches_hold_time():
    p0 = Plan(PlanConfig(handoffs=3, seed=3))
    p60 = Plan(PlanConfig(handoffs=3, seed=3, min_gap_s=60.0))
    gaps = [b.t_h - a.t_h for a, b in zip(p60.handoffs, p60.handoffs[1:])]
    assert all(g / FS >= 60.0 for g in gaps)
    assert p60.hold_bars > p0.hold_bars
    # Tracks kommen aus einem kleinen Pool (kein Speicher für hunderte Tracks)
    big = Plan(PlanConfig(handoffs=30, seed=1, min_gap_s=60.0))
    assert len(big.tracks) == 91
    for t in big.tracks:
        big.track_of(t)
    assert len(big._cache) == 8
    assert big.track_of(big.tracks[0]) is not big.track_of(big.tracks[1])
