"""Erweiterungen am Mixer für den Live-Betrieb: Folger-Modus (Stereo) und Umlabeln der Zeitachse."""

import numpy as np
import pytest

from b2b_relay import RelayConfig, RelayMixer

from conftest import FS
from test_mixer import delayed

CFG = dict(buffer_s=3.0, target_buffer_s=3.0, min_buffer_s=1.0, search_max_s=3.0)
LAG = int(0.2 * FS)


def _pair():
    m = RelayMixer(RelayConfig(**CFG))
    s = RelayMixer(RelayConfig(**CFG))
    s.follow(m)
    for x in (m, s):
        x.register_source("A")
        x.register_source("B")
    m.set_active("A")
    return m, s


def _drive(m, s, feeds_m, feeds_s, total, block=2048, hook=None):
    om, os_ = [], []
    for k in range(0, total - block + 1, block):
        om.append(m.process({n: f[k:k + block] for n, f in feeds_m.items()}))
        os_.append(s.process({n: f[k:k + block] for n, f in feeds_s.items()}))
        if hook:
            hook(k, m)
    return np.concatenate(om), np.concatenate(os_)


def test_follower_matches_master_exactly_through_handoff_and_rate_change(music):
    total = 40 * FS
    a, b = music[:total], delayed(music, LAG, total)
    m, s = _pair()
    done = []

    def hook(k, mix):
        if k > 15 * FS and not done:
            res = mix.request_handoff("B")
            done.append(res.accepted)

    # derselbe Inhalt auf beiden Kanälen: Folger muss bit-genau dasselbe liefern
    om, os_ = _drive(m, s, {"A": a, "B": b}, {"A": a, "B": b}, total, hook=hook)
    assert done == [True]
    assert m.status.active == "B"
    np.testing.assert_array_equal(om, os_)


def test_follower_carries_independent_side_signal(music):
    """Die Seite wird mit denselben Lags und Umschaltpunkten wie die Mitte verarbeitet."""
    total = 30 * FS
    a, b = music[:total], delayed(music, LAG, total)
    m, s = _pair()
    side_a, side_b = 0.3 * a[::-1].copy(), 0.3 * b[::-1].copy()
    done = []

    def hook(k, mix):
        if k > 12 * FS and not done:
            done.append(mix.request_handoff("B").accepted)

    om, os_ = _drive(m, s, {"A": a, "B": b}, {"A": side_a, "B": side_b}, total, hook=hook)
    assert done == [True]
    # Das Seiten-Programm ist 0 vor dem Start und hat danach Energie, die zu side_* gehört
    assert not os_[: int(3.0 * FS)].any()
    assert np.sqrt(np.mean(os_[int(5 * FS):] ** 2)) > 0.01


def test_retime_does_not_change_audio(music):
    total = 40 * FS
    a, b = music[:total], delayed(music, LAG, total)

    def run(shift_at):
        m = RelayMixer(RelayConfig(**CFG))
        for n in "AB":
            m.register_source(n)
        m.set_active("A")
        if shift_at is not None:
            m.retime(-480000)
        outs, flag = [], []
        for k in range(0, total - 2048 + 1, 2048):
            outs.append(m.process({"A": a[k:k + 2048], "B": b[k:k + 2048]}))
            if shift_at is not None and k in (10 * 2048, 200 * 2048):
                m.retime(37 if k == 10 * 2048 else -91)
            if k > 18 * FS and not flag:
                flag.append(m.request_handoff("B").accepted)
        assert flag == [True]
        return np.concatenate(outs), m

    ref, m0 = run(None)
    shifted, m1 = run(1)
    np.testing.assert_allclose(shifted, ref, atol=1e-5)
    assert m1.status.active == m0.status.active == "B"
    # Die Achse hat sich um genau die Summe verschoben, Rate und Puffer sind gleich
    assert abs(m1.buffer_s - m0.buffer_s) < 1e-6
    assert m1.live_edge - m0.live_edge == -480000 + 37 - 91


def test_retime_follower_rejected():
    m, s = _pair()
    with pytest.raises(RuntimeError):
        s.retime(5)
