import tracemalloc

import numpy as np

from b2b_relay.live.ring import AudioRing
from b2b_relay.live.wav import WavWriter, read_wav


def test_ring_roundtrip_wraparound():
    r = AudioRing(2, 1000)
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, 5000)).astype(np.float32)
    out = np.zeros((2, 5000), np.float32)
    w = rd = 0
    while rd < 5000:
        n = min(300, 5000 - w, r.free)
        if n:
            w += r.write((data[0, w:w + n], data[1, w:w + n]), n)
        m = r.read((out[0, rd:rd + 250], out[1, rd:rd + 250]), 250)
        rd += m
    assert np.array_equal(out, data)
    assert r.dropped == 0


def test_ring_overflow_drops_new_and_counts():
    r = AudioRing(1, 100)
    x = np.arange(80, dtype=np.float32)
    assert r.write((x,), 80) == 80
    assert r.write((x,), 80) == 20           # nur 20 Platz
    assert r.dropped == 60
    out = np.zeros(100, np.float32)
    assert r.read((out,), 100) == 100
    assert np.array_equal(out[:80], x) and np.array_equal(out[80:], x[:20])


def test_ring_underrun_pads_with_silence():
    r = AudioRing(1, 100)
    r.write((np.ones(10, np.float32),), 10)
    out = np.full(32, 7.0, np.float32)
    assert r.read_padded((out,), 32) == 10
    assert out[:10].tolist() == [1.0] * 10 and not out[10:].any()
    assert r.short == 1


def test_callback_path_allocates_no_array_data():
    """Kopieren im Callback-Pfad legt keine Datenpuffer an: 1000 Zyklen mit 4096 Frames × 2 Kanäle
    (je 16 KiB) bleiben unter ein paar KiB Spitzenverbrauch; ein Neuanlegen je Block wäre ≥ 16 KiB."""
    r = AudioRing(2, 1 << 14)
    src = (np.ones(4096, np.float32), np.ones(4096, np.float32))
    dst = (np.zeros(4096, np.float32), np.zeros(4096, np.float32))
    for _ in range(50):                         # Einschwingen (Caches, erste Ansichten)
        r.write(src, 4096)
        r.read(dst, 4096)
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    tracemalloc.reset_peak()
    cur0 = tracemalloc.get_traced_memory()[0]
    for _ in range(1000):
        r.write(src, 4096)
        r.read_padded(dst, 4096)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak - cur0 < 4096, f"Spitze {peak - cur0} B"
    assert cur - cur0 < 1024
    del base


def test_wav_roundtrip(tmp_path):
    for fmt, tol in (("float32", 0.0), ("pcm16", 1 / 32768)):
        p = tmp_path / f"{fmt}.wav"
        w = WavWriter(p, 48000, 2, fmt)
        t = np.arange(4800) / 48000
        l, rr = np.sin(2 * np.pi * 440 * t).astype(np.float32), 0.5 * np.cos(2 * np.pi * 220 * t).astype(np.float32)
        w.write((l[:2000], rr[:2000]))
        w.write((l[2000:], rr[2000:]))
        w.close()
        fs, a = read_wav(p)
        assert fs == 48000 and a.shape == (4800, 2)
        assert np.max(np.abs(a[:, 0] - l)) <= tol + 1e-7
