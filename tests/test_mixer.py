import numpy as np
import pytest

from b2b_relay import RelayConfig, RelayMixer
from b2b_relay.sim.analysis import find_gaps
from b2b_relay.sim.synth import Track
from b2b_relay.timeline import frac_read

from conftest import FS, run_mixer


def delayed(music, lag_samples, total):
    return frac_read(music, np.arange(total) - lag_samples).astype(np.float32)


def make(cfg=None, hint_b=None):
    m = RelayMixer(cfg or RelayConfig())
    m.register_source("A")
    m.register_source("B", rtt_hint_s=hint_b)
    m.set_active("A")
    return m


def test_no_source_gives_silence_and_flag():
    m = RelayMixer()
    out = m.process({})
    assert out.shape == (512,) and not out.any()
    assert m.status.no_source
    m.register_source("A")
    out = m.process({"A": np.ones(512, dtype=np.float32)})
    assert not out.any() and m.status.no_source and m.status.active is None


def test_output_is_program_delayed_by_buffer(music):
    total = 12 * FS
    m = make()
    out = run_mixer(m, {"A": music[:total], "B": np.zeros(total, dtype=np.float32)}, total)
    D = int(2.0 * FS)
    assert not out[:D].any()
    np.testing.assert_allclose(out[D:len(out)], music[:len(out) - D], atol=1e-6)


def test_handoff_is_seamless_and_estimate_is_exact(music):
    total = 40 * FS
    lag = 0.2503 * FS + 0.37                      # bewusst krumm, mit Bruchteil
    feeds = {"A": music[:total], "B": delayed(music, lag, total)}
    m = make()
    _seamless(m, music, feeds, total, lag, tol=0.25)


def test_handoff_is_exact_for_integer_lag(music):
    total = 40 * FS
    feeds = {"A": music[:total], "B": delayed(music, 12000, total)}
    _seamless(make(), music, feeds, total, 12000, tol=1e-4)


def _seamless(m, music, feeds, total, lag, tol):
    res = {}

    def hook(k, mix):
        if k == 25 * FS // 512 * 512:
            res["r"] = mix.request_handoff("B")
    out = run_mixer(m, feeds, total, hook=hook)
    r = res["r"]
    assert r.accepted, r
    info = m.source_info("B")
    assert info.locked and abs(info.lag_s * FS - lag) < 1.0          # < 21 µs
    assert m.status.active == "B" and m.status.previous == "A"
    D = int(2.0 * FS)
    ref = music[:len(out) - D]
    err = np.abs(out[D:] - ref)
    assert err.max() < tol, err.max()
    # keine Unstetigkeit beim Umschalten: größter Sprung nicht größer als im Original
    assert np.abs(np.diff(out[D:])).max() <= np.abs(np.diff(ref)).max() * 1.1
    assert not find_gaps(out[D:], FS)


def test_handoff_at_given_time_and_out_of_order(music):
    total = 30 * FS
    feeds = {"A": music[:total], "B": delayed(music, 0.15 * FS, total)}
    m = make()
    run_mixer(m, feeds, 20 * FS, start=0)
    r = m.request_handoff("B", at=(m.live_edge / FS) + 3.0)
    assert r.accepted and abs(r.t_switch / FS - (m.live_edge / FS + 3.0)) < 0.01
    again = m.request_handoff("A", at=(m.live_edge / FS) + 1.0)     # früher als die vorgemerkte
    assert not again.accepted and again.reason in ("out_of_order", "not_locked", "already_active")
    late = m.request_handoff("A", at=0.5)
    assert not late.accepted
    assert m.status.active == "A"                                    # noch nicht wirksam
    for _ in range(int(4 * FS / 512) + 1):
        m.process({"A": np.zeros(512, dtype=np.float32), "B": np.zeros(512, dtype=np.float32)})
    assert m.status.active == "B"


def test_handoff_refused_when_rtt_exceeds_buffer_minus_reserve():
    cfg = RelayConfig(buffer_s=1.0, reserve_s=0.2)
    m = make(cfg, hint_b=0.9)
    r = m.request_handoff("B")
    assert not r.accepted and r.reason == "rtt_exceeds_buffer"
    assert "900 ms" in r.message and "1000 ms" in r.message and "200 ms" in r.message
    m2 = make(cfg, hint_b=0.79)                  # 0,79 + 0,2 ≤ 1,0 → Puffer reicht, aber keine Messung
    assert m2.request_handoff("B").reason == "not_locked"
    m3 = make(cfg, hint_b=0.81)
    assert m3.request_handoff("B").reason == "rtt_exceeds_buffer"
    assert m3.status.active == "A"


def test_handoff_refused_with_measured_lag_beyond_budget(music):
    total = 30 * FS
    cfg = RelayConfig(buffer_s=1.0, reserve_s=0.2)
    feeds = {"A": music[:total], "B": delayed(music, 0.9 * FS, total)}
    m = make(cfg)
    run_mixer(m, feeds, total)
    assert m.source_info("B").locked
    assert abs(m.source_info("B").lag_s - 0.9) < 1e-3
    r = m.request_handoff("B")
    assert not r.accepted and r.reason == "rtt_exceeds_buffer"
    assert m.status.active == "A"


def test_handoff_accepted_exactly_within_budget(music):
    total = 25 * FS
    cfg = RelayConfig(buffer_s=1.0, reserve_s=0.2)
    feeds = {"A": music[:total], "B": delayed(music, 0.78 * FS, total)}
    m = make(cfg)
    run_mixer(m, feeds, total)
    assert m.request_handoff("B").accepted


def test_unknown_and_double_handoff():
    m = make()
    assert m.request_handoff("X").reason == "unknown_source"
    assert m.request_handoff("A").reason == "already_active"


def test_fallback_when_active_source_drops_out_for_500ms(music):
    total = 40 * FS
    lag = 0.12 * FS
    b = delayed(music, lag, total)
    t_out = 30 * FS
    b_fail = b.copy()
    b_fail[t_out:t_out + FS // 2] = 0.0                     # 500 ms Ausfall der aktiven Quelle
    feeds = {"A": music[:total], "B": b_fail}
    m = make()
    state = {}

    def hook(k, mix):
        if k == 22 * FS // 512 * 512:
            state["r"] = mix.request_handoff("B")
    out = run_mixer(m, feeds, total, hook=hook)
    assert state["r"].accepted
    kinds = [e.kind for e in m.events]
    assert "fallback" in kinds
    fb = next(e for e in m.events if e.kind == "fallback")
    assert fb.detail["source"] == "B" and fb.detail["to"] == "A"
    assert m.status.active == "A" and m.status.fallback
    D = int(2.0 * FS)
    # Programm bleibt durchgehend hörbar und stimmt mit dem Original überein, auch im Ausfall
    t0 = int(t_out - lag)
    seg = out[D + t0:D + t0 + FS // 2 + 4800]
    ref = music[t0:t0 + FS // 2 + 4800]
    assert np.abs(seg - ref).max() < 0.05
    assert not find_gaps(out[D:], FS)


def test_silence_without_fallback_candidate_sets_flag(music):
    total = 20 * FS
    a = music[:total].copy()
    a[12 * FS:] = 0.0
    m = make()
    run_mixer(m, {"A": a, "B": np.zeros(total, dtype=np.float32)}, total)
    assert m.status.silent and m.status.active == "A"


def test_estimate_survives_when_program_part_fades_out(music):
    other = Track(seed=21, bars=60).mix
    total = 60 * FS
    lag = 0.31 * FS + 0.2
    echo = delayed(music, lag, total)
    n = np.arange(total)
    # 15–25 s: Programmanteil sinkt auf 0; der DJ spielt ab 15 s seinen eigenen Track dazu
    g = np.clip(1.0 - (n - 15 * FS) / (10.0 * FS), 0.0, 1.0)
    g = g ** 2
    b = (echo * g + 0.6 * other[:total] * (n > 15 * FS)).astype(np.float32)
    m = make()
    marks = {}

    def hook(k, mix):
        if k // 512 == 14 * FS // 512:
            marks["before"] = mix.source_info("B").lag_s
        if k // 512 == 58 * FS // 512:
            marks["after"] = mix.source_info("B").lag_s
    run_mixer(m, {"A": music[:total], "B": b}, total, hook=hook)
    assert abs(marks["before"] * FS - lag) < 1.0
    info = m.source_info("B")
    assert info.locked
    assert abs(marks["after"] * FS - lag) < 5.0       # gehalten (kein Sprung zum fremden Track)
    assert m.request_handoff("B", at=None).reason in ("ok", "estimate_stale", "target_silent")


def test_estimate_follows_when_echo_is_26db_down(music):
    other = Track(seed=22, bars=60).mix
    total = 30 * FS
    lag = 0.4 * FS + 0.6
    b = (delayed(music, lag, total) * 10 ** (-26 / 20) + 0.5 * other[:total]).astype(np.float32)
    m = make()
    run_mixer(m, {"A": music[:total], "B": b}, total)
    info = m.source_info("B")
    assert info.locked and abs(info.lag_s * FS - lag) < 2.0


@pytest.mark.slow
def test_estimate_follows_100ppm_drift_over_10_minutes():
    secs = 600
    total = secs * FS
    music = Track(seed=9, bars=int(secs / 1.3793) + 8).mix
    eps = 100e-6
    lag0 = 0.2 * FS
    cfg = RelayConfig(est_window_s=3.0, est_hop_s=1.0, buffer_s=2.0)
    m = make(cfg)
    block = 2048
    worst = 0.0
    for k in range(0, total - block, block):
        w = np.arange(k, k + block, dtype=np.float64)
        lag = lag0 + eps * w
        b = frac_read(music, w - lag).astype(np.float32)
        m.process({"A": music[k:k + block], "B": b})
        if k > 20 * FS and (k // block) % 20 == 0:
            info = m.source_info("B")
            assert info.locked
            true = lag0 + eps * (k + block)
            worst = max(worst, abs(m.lag_of("B") - true))
    assert worst < 0.1e-3 * FS, f"max. Fehler {worst / FS * 1e3:.3f} ms"
    assert abs(m.source_info("B").drift_ppm - 100.0) < 5.0
