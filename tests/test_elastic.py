"""Elastischer Ausgang: Regler, Varispeed-Stufe, Übergabe-Freigabe nach Puffer, Langzeitlauf."""

import math
import os

import numpy as np
import pytest

from b2b_relay import RelayConfig, RelayMixer
from b2b_relay.control import BufferController
from b2b_relay.sim.scenario import OutputMap
from b2b_relay.varispeed import VarispeedReader, sinc_table

from conftest import FS
from test_mixer import delayed


# ---------------------------------------------------------------- Regler

def run_controller(ctl, b0, seconds, dt=0.01, kicks=()):
    """B-Dynamik: dB/dt = −dev; ``kicks`` = [(t, ΔB)] (Übergaben). Liefert Spuren."""
    b, t = b0, 0.0
    ks = sorted(kicks)
    tr = {"b": [], "dev": []}
    while t < seconds:
        while ks and ks[0][0] <= t:
            b += ks.pop(0)[1]
        dev = ctl.step(b, dt)
        b -= dev * dt
        t += dt
        tr["b"].append(b)
        tr["dev"].append(dev)
    return np.array(tr["b"]), np.array(tr["dev"])


def test_controller_converges_within_limits_without_overshoot():
    ctl = BufferController(3.0, 0.1, 0.005, 0.001)
    b, dev = run_controller(ctl, 2.0, 400)
    assert np.abs(dev).max() <= 0.005 + 1e-12
    assert np.abs(np.diff(dev)).max() <= 0.001 * 0.01 + 1e-12          # Slew eingehalten
    assert b.max() <= 3.0 + 1e-6                                        # kein Überschwingen
    assert abs(b[-1] - 3.0) <= 0.05 + 1e-3 and dev[-1] == 0.0           # Ruhe nahe Ziel
    # von oben genauso
    ctl = BufferController(3.0, 0.1, 0.005, 0.001)
    b, dev = run_controller(ctl, 3.6, 400)
    assert b.min() >= 3.0 - 0.05 - 1e-3 and abs(b[-1] - 3.0) <= 0.05 + 1e-3
    assert dev.max() <= 0.005 + 1e-12 and dev.min() >= -1e-12


def test_controller_deadband_holds_rate_and_does_not_oscillate():
    ctl = BufferController(3.0, 0.1, 0.005, 0.001)
    rng = np.random.default_rng(0)
    for _ in range(5000):                       # B wackelt ±0,09 s um das Ziel: nichts passiert
        ctl.step(3.0 + rng.uniform(-0.09, 0.09), 0.01)
    assert ctl.dev == 0.0 and not ctl.engaged
    # eine Übergabe wirft B heraus: Regelung läuft einmal hin, ohne Vorzeichenwechsel
    b, dev = run_controller(ctl, 2.75, 300)
    assert dev.max() <= 1e-12                                            # nur „langsamer“
    signs = np.sign(dev[np.abs(dev) > 1e-9])
    assert (signs == signs[0]).all()
    assert abs(b[-1] - 3.0) <= 0.051


def test_controller_time_to_matches_forward_run():
    ctl = BufferController(3.0, 0.1, 0.005, 0.001)
    eta = ctl.time_to(2.0, 2.5)
    b, dev = run_controller(BufferController(3.0, 0.1, 0.005, 0.001), 2.0, 200, dt=0.01)
    actual = (np.argmax(b >= 2.5) + 1) * 0.01
    assert eta == pytest.approx(actual, rel=0.02)
    assert ctl.time_to(2.0, 2.0) == 0.0
    assert ctl.time_to(2.0, 3.2) is None            # liegt jenseits des Ruhepunkts


# ---------------------------------------------------------------- Varispeed-Stufe

def test_varispeed_is_identity_at_unity_rate_and_flat_in_band():
    s = np.random.default_rng(0).standard_normal(4000)
    rd = VarispeedReader(100.0)
    out = rd.render(lambda a, n: s[a:a + n], 512)
    np.testing.assert_array_equal(out, s[100:612])
    assert rd.position == 612.0
    tab = sinc_table()
    for ph in (0.2, 0.5, 0.8):
        row = tab.rows[int(ph * tab.phases)]
        for f in (0.1, 0.25, 0.4):              # 4,8 / 12 / 19,2 kHz bei 48 kHz
            h = abs(np.sum(row * np.exp(-2j * np.pi * f * tab.k)))
            assert abs(20 * math.log10(h)) < 0.01


def test_output_map_round_trip():
    n = 512
    rng = np.random.default_rng(1)
    r0 = 1 + rng.uniform(-0.005, 0.005, 50)
    r1 = np.append(r0[1:], 1.0)
    p0 = np.concatenate(([-96000.0], -96000.0 + np.cumsum(n * 0.5 * (r0 + r1))[:-1]))
    om = OutputMap(n, p0, r0, r1)
    j = np.array([0.0, 5.5, 511.0, 512.0, 20000.25])
    np.testing.assert_allclose(om.j_of(om.t_of(j)), j, atol=1e-6)


def sine(freq, n0, n, fs=FS, amp=0.5):
    return amp * np.sin(2 * np.pi * freq * (np.arange(n0, n0 + n)) / fs)


def test_rate_changes_are_click_free_and_phase_exact():
    """Hart gesprungene Raten (0,995 ↔ 1,005 zwischen Blöcken) und Rampen: Ausgang = Sinus an
    den gelesenen Positionen, größter Sample-Sprung im Rahmen der Signaländerung."""
    f, amp = 2000.0, 0.5
    x = amp * np.sin(2 * np.pi * f * np.arange(400000) / FS)
    rd = VarispeedReader(1000.0)
    read = lambda a, n: x[a:a + n]
    n = 512
    outs, pos = [], []
    profile = [(1.0, 1.0)] * 3 + [(0.995, 0.995)] * 3 + [(1.005, 1.005)] * 3 + [(1.005, 0.995)] * 2 \
        + [(0.995, 1.0)] * 2 + [(1.0, 1.0)] * 2
    for r0, r1 in profile:
        p0 = rd.position
        out = rd.render(read, n, r0, r1)
        i = np.arange(n)
        pos.append(p0 + r0 * i + (r1 - r0) * i * i / (2.0 * n))
        outs.append(out)
    out, pos = np.concatenate(outs), np.concatenate(pos)
    expect = amp * np.sin(2 * np.pi * f * pos / FS)
    assert np.abs(out - expect).max() < 1e-4        # besser als −74 dB
    natural = amp * 2 * np.pi * f / FS * 1.005
    assert np.abs(np.diff(out)).max() <= natural * 1.001


def test_mixer_output_follows_exact_positions_while_buffer_fills():
    """Elastischer Mixer, Quelle = Sinus über der absoluten Zeit: jedes Ausgabe-Sample liegt
    auf der Sinuskurve an der vom Mixer gemeldeten Leseposition; Rampe 1 → 0,995 → 1."""
    f, amp = 1500.0, 0.5
    cfg = RelayConfig(buffer_s=2.0, target_buffer_s=2.4)
    m = RelayMixer(cfg)
    m.register_source("A")
    m.set_active("A")
    n, total = 512, 120 * FS
    x = sine(f, 0, total + n)
    err, jump, last, rates = 0.0, 0.0, None, []
    for k in range(0, total, n):
        out = m.process({"A": x[k:k + n].astype(np.float32)})
        lb = m.last_block
        i = np.arange(n)
        pos = lb.pos + lb.r0 * i + (lb.r1 - lb.r0) * i * i / (2.0 * n)
        if lb.pos > 2000:
            err = max(err, float(np.abs(out - sine_at(f, pos, amp)).max()))
            if last is not None:
                jump = max(jump, abs(float(out[0]) - last))
            jump = max(jump, float(np.abs(np.diff(out)).max()))
        last = float(out[-1])
        rates.append(lb.r0)
    assert err < 2e-4                       # float32-Eingang
    assert jump <= amp * 2 * np.pi * f / FS * 1.005 + 1e-4
    assert min(rates) == pytest.approx(0.995) and abs(rates[-1] - 1.0) < 1e-9
    assert m.buffer_s == pytest.approx(2.4, abs=0.06)


def sine_at(f, pos, amp):
    return amp * np.sin(2 * np.pi * f * pos / FS)


# ---------------------------------------------------------------- Status und Freigabe

def test_status_reports_buffer_rate_and_equivalent_bpm():
    m = RelayMixer(RelayConfig(buffer_s=2.0, target_buffer_s=3.0))
    m.register_source("A")
    m.register_source("B", rtt_hint_s=0.25)
    assert m.status.buffer.rate == 1.0 and m.status.buffer.rate_bpm == 174.0
    m.set_active("A")
    blk = np.zeros(512, dtype=np.float32)
    for _ in range(int(20 * FS / 512)):
        m.process({"A": blk})
    b = m.status.buffer
    assert b.enabled and b.target_s == 3.0 and b.min_s == 1.0
    assert 0.99 < b.rate < 1.0 and b.rate_bpm == pytest.approx(174.0 * b.rate)
    assert b.buffer_s > 2.0 and b.lag_s == 0.0
    assert b.output_delay_s == pytest.approx(b.buffer_s)            # noch kein Lag
    assert b.release and b.wait_s == 0.0 and b.handoff["B"].ok
    assert b.handoff["B"].buffer_after_s == pytest.approx(b.buffer_s - 0.25, abs=1e-6)
    assert b.eta_target_s is not None and 60 < b.eta_target_s < 200
    # volle Rate nach langer Zeit: 173,13 BPM
    for _ in range(int(20 * FS / 512)):
        m.process({"A": blk})
    assert m.status.buffer.rate_bpm == pytest.approx(174.0 * 0.995, abs=0.005)
    assert round(m.status.buffer.rate_bpm, 1) == 173.1
    # ohne Regelung: Rate 1, nichts zu melden
    off = RelayMixer(RelayConfig())
    off.register_source("A")
    off.set_active("A")
    off.process({"A": blk})
    assert not off.status.buffer.enabled and off.status.buffer.rate == 1.0


def test_handoff_refused_when_buffer_too_low_then_released_with_plausible_wait(music):
    total = 70 * FS
    lag = int(0.25 * FS)
    feeds = {"A": music[:total], "B": delayed(music, lag, total)}
    cfg = RelayConfig(buffer_s=1.1, target_buffer_s=3.0, max_dev=0.02, max_slew=0.002)
    m = RelayMixer(cfg)
    m.register_source("A")
    m.register_source("B")
    m.set_active("A")
    n = 512
    first = None
    done_at = None
    for k in range(0, total - n, n):
        m.process({name: f[k:k + n] for name, f in feeds.items()})
        t = (k + n) / FS
        if first is None and t > 8.0 and m.status.sources["B"].locked:
            res = m.check_handoff("B")
            if not res.accepted:
                assert res.reason == "buffer_too_low"
                assert res.wait_s is not None and res.wait_s > 3.0
                assert "voraussichtlich frei in" in res.message
                first = (t, res.wait_s, res.buffer_after_s)
                assert res.buffer_after_s < cfg.min_buffer_s
        elif first is not None:
            res = m.check_handoff("B")
            if res.accepted:
                done_at = t
                break
    assert first is not None and done_at is not None
    waited = done_at - first[0]
    assert waited == pytest.approx(first[1], rel=0.2)
    # freigegeben: scharf stellen und prüfen, dass der Puffer nach der Wirkung ≥ B_min bleibt
    res = m.request_handoff("B")
    assert res.accepted and res.buffer_after_s >= cfg.min_buffer_s
    low = 99.0
    for k in range(int(done_at * FS) // n * n, total - n, n):
        m.process({name: f[k:k + n] for name, f in feeds.items()})
        low = min(low, m.buffer_s)
        if m.status.active == "B" and (k / FS) > done_at + 6:
            break
    assert m.status.active == "B"
    assert low >= cfg.min_buffer_s - 0.02


# ---------------------------------------------------------------- Langzeit

def rss_mb() -> float:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


@pytest.mark.skipif(not os.path.exists("/proc/self/statm"), reason="braucht /proc")
def test_three_hours_without_memory_growth_or_precision_loss():
    """3 h simulierte Zeit (vereinfacht: 2 kHz Abtastrate, Lags von außen gesetzt, keine
    Audio-Schätzung): alle 60 s eine Übergabe A↔B (RTT 40/250 ms) → der absolute Lag und die
    Ausgabeverzögerung wachsen auf über 30 s; Ringe und Listen bleiben gleich groß, die
    Ausgabe liegt am Ende noch auf ~1e-4 an der Sinuskurve der gemeldeten Leseposition."""
    fs = 2000
    f, amp = 97.0, 0.5
    cfg = RelayConfig(sample_rate=fs, buffer_s=2.0, target_buffer_s=3.0, est_hop_s=1e9)
    m = RelayMixer(cfg)
    m.register_source("A")
    m.register_source("B")
    m.set_active("A")
    n = 500
    hours3 = 3 * 3600 * fs
    rtt = {"A": 0.040, "B": 0.250}
    lag_of = {"A": 0, "B": 0}                    # Lag der Quelle in Samples (ganzzahlig)
    active, accepted, refused = "A", 0, []
    lows, rss_1h, caps = [], None, None
    next_ho = 60 * fs
    err_tail = 0.0

    def feed(name, w0):
        w = np.arange(w0, w0 + n, dtype=np.float64) - lag_of[name]
        return (amp * np.sin(2 * np.pi * f * w / fs)).astype(np.float32)

    for w0 in range(0, hours3, n):
        if w0 >= next_ho:
            tgt = "B" if active == "A" else "A"
            lag_of[tgt] = lag_of[active] + int(round(rtt[tgt] * fs))
            m.set_lag(tgt, lag_of[tgt] / fs)
            res = m.request_handoff(tgt)
            if res.accepted:
                accepted += 1
                active = tgt
            else:
                refused.append((w0 / fs, res.reason))
            next_ho += 60 * fs
        out = m.process({"A": feed("A", w0), "B": feed("B", w0)})
        lb = m.last_block
        lows.append(lb.buffer_s)
        if w0 > hours3 - 20 * fs or w0 % (600 * fs) == 0:
            i = np.arange(n)
            pos = lb.pos + lb.r0 * i + (lb.r1 - lb.r0) * i * i / (2.0 * n)
            if w0 > 40 * fs:
                e = float(np.abs(out - sine_at_fs(f, pos, amp, fs)).max())
                err_tail = max(err_tail, e)
                assert e < 2e-4, (w0 / fs, e)
        if w0 == 3600 * fs:
            rss_1h = rss_mb()
            caps = (len(m._segments), len(m.events), {k: s.ring.capacity for k, s in m._sources.items()})
    assert not refused and accepted == 179
    assert min(lows[200:]) >= cfg.min_buffer_s              # Puffer nie unter B_min (nach dem Start)
    assert max(lows) <= 3.2
    assert abs(lows[-1] - 3.0) <= 0.15
    st = m.status.buffer
    assert st.output_delay_s > 28 and st.lag_s > 25         # der absolute Lag ist riesig geworden
    assert abs(st.rate - 1.0) <= 0.005 + 1e-9
    # kein Wachstum: Segmente, Ringe, Ereignisse beschränkt; Prozessspeicher stabil
    assert len(m._segments) <= 4
    assert {k: s.ring.capacity for k, s in m._sources.items()} == caps[2]
    assert len(m.events) <= 4096
    assert max(s.ring.capacity for s in m._sources.values()) <= int(cfg.ring_s * fs)
    assert cfg.ring_s < 15
    assert rss_mb() - rss_1h < 15, (rss_1h, rss_mb())
    assert err_tail < 2e-4


def sine_at_fs(f, pos, amp, fs):
    return amp * np.sin(2 * np.pi * f * pos / fs)
