import numpy as np

from b2b_relay.rtt import LagTracker, gcc_phat
from b2b_relay.timeline import frac_read

FS = 48000


def _case(music, d, g=1.0, other=None, other_gain=0.0, x0=20 * FS, w=4 * FS, lo=0, hi=96000):
    n = np.arange(x0, x0 + w)
    x = g * frac_read(music, n - d)
    if other is not None:
        x = x + other_gain * other[x0:x0 + w]
    p = music[x0 - hi:x0 + w - lo]
    return x, p, hi - lo


def test_gcc_phat_integer_and_fractional(music):
    for d in (4800.0, 12345.0, 12345.4, 80000.7):
        x, p, ms = _case(music.astype(np.float64), d)
        m = gcc_phat(x, p, ms)
        assert m is not None and abs(m.delay - d) < 0.5 and m.confidence > 0.9


def test_gcc_phat_search_window_offset(music):
    d = 30000.3
    x, p, ms = _case(music.astype(np.float64), d, lo=29000, hi=31000)
    m = gcc_phat(x, p, ms)
    assert abs(29000 + m.delay - d) < 0.5


def test_gcc_phat_weak_echo_with_loud_other_track(music):
    from b2b_relay.sim.synth import Track
    other = Track(seed=77, bars=60).mix.astype(np.float64)     # anderer Track, gleiches Tempo
    for gain_db in (-20, -26):
        x, p, ms = _case(music.astype(np.float64), 22222.2, g=10 ** (gain_db / 20), other=other,
                         other_gain=1.0)
        m = gcc_phat(x, p, ms)
        assert abs(m.delay - 22222.2) < 1.0 and m.confidence >= 0.5, (gain_db, m)


def test_gcc_phat_no_echo_low_confidence(music):
    from b2b_relay.sim.synth import Track
    other = Track(seed=78, bars=60).mix.astype(np.float64)
    x, p, ms = _case(music.astype(np.float64), 10000.0, g=0.0, other=other, other_gain=1.0)
    m = gcc_phat(x, p, ms)
    assert m is None or m.confidence < 0.5


def test_gcc_phat_silence_returns_none():
    assert gcc_phat(np.zeros(5000), np.zeros(6000), 1000) is None


def test_tracker_locks_follows_drift_and_holds():
    tr = LagTracker()
    rng = np.random.default_rng(3)
    rate = 100e-6
    for k in range(60):                      # 0,5 s Abstand, 30 s
        t = k * 24000.0
        tr.update(t, 9600.0 + rate * t + rng.normal(0, 2.0), 0.9)
    assert tr.locked
    assert abs(tr.rate - rate) < 8e-6
    t_end = 59 * 24000.0
    assert abs(tr.predict(t_end) - (9600.0 + rate * t_end)) < 5.0
    # niedrige Konfidenz ändert nichts: letzte Schätzung wird gehalten und fortgeschrieben
    before = (tr.lag, tr.rate)
    assert not tr.update(t_end + 24000, 0.0, 0.1)
    assert (tr.lag, tr.rate) == before
    assert abs(tr.predict(t_end + 48000 * 20) - (9600.0 + rate * (t_end + 48000 * 20))) < 100.0


def test_tracker_rejects_single_outlier_but_reacquires_after_three():
    tr = LagTracker()
    for k in range(10):
        tr.update(k * 24000.0, 9600.0, 0.9)
    assert not tr.update(10 * 24000.0, 20000.0, 0.95)
    assert abs(tr.lag - 9600.0) < 1.0
    assert not tr.update(11 * 24000.0, 20000.0, 0.95)
    assert tr.update(12 * 24000.0, 20000.0, 0.95)
    assert abs(tr.lag - 20000.0) < 1.0
