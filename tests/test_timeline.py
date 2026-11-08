import numpy as np

from b2b_relay.timeline import SampleRing, frac_read


def test_ring_write_read_wrap():
    r = SampleRing(100)
    r.write(np.arange(70, dtype=np.float32))
    r.write(np.arange(70, 150, dtype=np.float32))     # läuft über den Rand
    assert r.end == 150 and r.start == 50
    np.testing.assert_array_equal(r.read(120, 10), np.arange(120, 130))
    np.testing.assert_array_equal(r.read(50, 100), np.arange(50, 150))


def test_ring_unavailable_is_zero():
    r = SampleRing(100)
    r.write(np.ones(30, dtype=np.float32))
    out = r.read(-5, 50)                 # vor dem Anfang und hinter dem Ende
    assert out[:5].sum() == 0 and out[5:35].sum() == 30 and out[35:].sum() == 0
    r.write(np.ones(200, dtype=np.float32))
    assert r.read(0, 10).sum() == 0      # überschrieben


def test_ring_write_larger_than_capacity():
    r = SampleRing(16)
    r.write(np.arange(40, dtype=np.float32))
    assert r.end == 40
    np.testing.assert_array_equal(r.read(24, 16), np.arange(24, 40))


def test_frac_read_exact_on_integers_and_accurate_between():
    n = np.arange(2000)
    x = np.sin(2 * np.pi * 440 * n / 48000.0)
    np.testing.assert_allclose(frac_read(x, np.arange(10, 100, dtype=float)), x[10:100], atol=1e-12)
    pos = np.arange(10, 1900) + 0.37
    ref = np.sin(2 * np.pi * 440 * pos / 48000.0)
    assert np.abs(frac_read(x, pos) - ref).max() < 1e-4


def test_ring_read_frac_matches_array():
    x = np.random.default_rng(0).standard_normal(500).astype(np.float32)
    r = SampleRing(300)
    r.write(x)
    pos = np.arange(300, 480) + 0.25
    np.testing.assert_allclose(r.read_frac(pos), frac_read(x, pos), atol=1e-6)
