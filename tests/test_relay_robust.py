"""Bausteine des Fixes S3904 einzeln: Prüfung der Einrastung, Slot-Sprünge, getrennte Messfenster,
Verschleierung von Aussetzern, Nachbildung des JackTrip-Jitterpuffers."""

import numpy as np

from b2b_relay.conceal import GapConcealer
from b2b_relay.mixer import RelayConfig, RelayMixer
from b2b_relay.rtt import LagTracker
from b2b_relay.sim.analysis import find_gaps_chunked
from b2b_relay.timeline import SampleRing
from conftest import run_mixer
from netsim import NetemUplink

FS = 48000
SLOT = 256
LIVE = dict(est_window_s=4.0, est_hop_s=0.5, max_rate_ppm=0.0, track_window_s=0.030, accept_s=0.012,
            fit_span_s=12.0, confirm_s=4.5, relock_s=2.0, verify_every_s=2.0, reacquire_narrow_s=12.0,
            search_max_s=3.0)     # wie live/engine.py


def _delayed(x: np.ndarray, d: int) -> np.ndarray:
    return np.concatenate([np.zeros(d, np.float32), x[:len(x) - d]])


def test_false_lock_is_rejected_and_the_true_lag_found(music):
    """Eingerastet auf einen falschen Lag (wie B in Run 13: 266,7 statt ≈ 70 ms): die Prüfung gegen den vollen
    Suchbereich verwirft die Einrastung, danach rastet die Quelle auf den wahren Lag ein. Ohne Prüfung bliebe
    die Schätzung für immer falsch (Nachführung sucht nur um die Vorhersage)."""
    true = 3360                                        # 70 ms
    m = RelayMixer(RelayConfig(buffer_s=2.0, **LIVE))
    m.register_source("A")
    m.register_source("B")
    m.set_active("A")
    feeds = {"A": music, "B": _delayed(music, true)}
    run_mixer(m, feeds, FS, block=2048)                # B hat Signal (sonst setzt der Signalbeginn zurück)
    m.set_lag("B", 0.2667)                             # falsche Einrastung
    run_mixer(m, feeds, 40 * FS, block=2048, start=FS)
    kinds = [e.kind for e in m.events]
    assert "lock_rejected" in kinds
    info = m.source_info("B")
    assert info.locked
    assert abs(m.lag_of("B") - m.lag_of("A") - true) < 1.0


def test_slot_step_of_a_waiting_source_is_followed(music):
    """Der Lag einer wartenden Quelle springt um einen JackTrip-Slot (Unterlauf des Jitterpuffers, 5,3 ms) und
    bleibt dort: nach dem Neu-Aufsetzen stimmt die Schätzung wieder (vorher: Maximum außerhalb ±3 ms, die
    Schätzung veraltet, Übergabe ``estimate_stale``)."""
    d0, d1 = 3360, 3360 + SLOT
    m = RelayMixer(RelayConfig(buffer_s=2.0, **LIVE))
    for n in "AB":
        m.register_source(n)
    m.set_active("A")
    b = np.concatenate([_delayed(music, d0)[:30 * FS], _delayed(music, d1)[30 * FS:]])
    feeds = {"A": music, "B": b}
    run_mixer(m, feeds, 30 * FS, block=2048)
    assert abs(m.lag_of("B") - m.lag_of("A") - d0) < 1.0
    run_mixer(m, feeds, 50 * FS, block=2048, start=30 * FS)
    assert abs(m.lag_of("B") - m.lag_of("A") - d1) < 1.0
    assert m.source_info("B").stale_s < 4.0                 # Freigabe verlangt < max_stale_s (8 s)


def test_lock_needs_measurements_from_separate_windows():
    t = LagTracker(gate=144.0, rate_limit=0.0, confirm_span=4.5 * FS, accept=576.0)
    assert not t.update(4.0 * FS, 3360.0, 0.9)
    assert not t.update(4.5 * FS, 3360.0, 0.9)           # stimmig, aber dasselbe Fenster
    assert not t.locked
    assert t.update(8.6 * FS, 3362.0, 0.9)
    assert t.locked and t.rate == 0.0


def test_mean_of_jittering_slot_states():
    """Lag zappelt zwischen zwei Slot-Zuständen (70 % / 30 %): die Schätzung ist der Mittelwert."""
    t = LagTracker(gate=144.0, rate_limit=0.0, accept=576.0, fit_span=12.0 * FS, fit_min_span=2.0 * FS)
    t.set_reference(3360.0, 0.0)
    rng = np.random.default_rng(3)
    for k in range(1, 60):
        t.update(k * 0.5 * FS, 3360.0 + SLOT * (rng.random() < 0.3), 0.9)
    assert abs(t.lag - (3360.0 + 0.3 * SLOT)) < 0.2 * SLOT


def test_gap_concealer_fills_dropouts_but_not_silence():
    x = (0.3 * np.sin(2 * np.pi * 220.0 * np.arange(20 * FS) / FS)).astype(np.float32)
    x[2 * FS:2 * FS + SLOT] = 0.0                          # ein Paket
    x[3 * FS - 100:3 * FS + 3 * SLOT - 100] = 0.0          # drei Pakete über eine Blockgrenze
    x[4 * FS + 2048 - SLOT:4 * FS + 2048] = 0.0            # endet genau an einer Blockgrenze
    x[10 * FS:12 * FS] = 0.0                               # Quelle verstummt (2 s): bleibt
    ring, gc = SampleRing(len(x) + 1), GapConcealer(int(0.1 * FS))
    for k in range(0, len(x), 2048):
        blk = x[k:k + 2048]
        ring.write(blk)
        gc.after_write(ring, k, blk)
    y = ring.read(0, len(x))
    gaps = find_gaps_chunked(y, FS, 5.0)
    assert gc.count == 3
    assert len(gaps) == 1 and gaps[0][0] == 10 * FS and abs(gaps[0][1] - 2 * FS) <= 1
    assert np.max(np.abs(np.diff(y[FS:9 * FS]))) < 0.05      # kein Knacks (Sinus 220 Hz: Schritt ≤ 0,009)


def test_concealed_feed_gives_program_without_gaps(music):
    a = music.copy()
    rng = np.random.default_rng(1)
    for s in rng.integers(5 * FS, 60 * FS, 40) // SLOT * SLOT:
        a[s:s + SLOT * int(rng.integers(1, 4))] = 0.0
    m = RelayMixer(RelayConfig(buffer_s=2.0, conceal_max_s=0.1, **LIVE))
    m.register_source("A")
    m.set_active("A")
    out = run_mixer(m, {"A": a}, 70 * FS, block=2048)
    assert find_gaps_chunked(out[8 * FS:], FS, 5.0) == []
    assert m.concealed["A"][0] >= 30


def test_netsim_jitter_buffer_reproduces_the_server_log():
    """Profile aus run.py: clean ohne Aussetzer (AutoQueue 3 wie im Server-Log), wan40 mit Aussetzern von
    einem Slot (Log: AutoQueue 4-5, Run 13: Lücken 5,1-5,3 ms), wan120 mit Läufen bis 6 Slots (Log: 9-15,
    Run 13: max 31,9 ms)."""
    def run(seconds=60, **kw):
        u = NetemUplink(seed=5, **kw)
        x = np.full(SLOT, 0.1, np.float32)
        y = np.concatenate([u.step(x, x)[0] for _ in range(seconds * FS // SLOT)])
        return find_gaps_chunked(y[8 * FS:], FS, 5.0), [q for _, q in u.jb.queue_log]
    g, q = run(delay_ms=0.5)
    assert g == [] and set(q) == {3}
    g, q = run(delay_ms=40, jitter_ms=5)
    assert len(g) > 5 and {n % SLOT for _, n in g} == {0} and 4 <= max(q) <= 6
    g, q = run(delay_ms=120, jitter_ms=20, loss=0.005, reorder=0.05, reorder_corr=0.5)
    assert len(g) > 20 and max(n for _, n in g) <= 8 * SLOT and 9 <= max(q) <= 16
