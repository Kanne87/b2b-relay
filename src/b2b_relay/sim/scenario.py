"""Szenario A→B→A… : Simulation durchlaufen lassen und je Übergabe vermessen."""

from __future__ import annotations

import time
import wave
from dataclasses import asdict, dataclass

import numpy as np

from ..mixer import RelayConfig, RelayMixer
from ..timeline import frac_read
from .analysis import find_gaps, find_gaps_chunked, kick_phase_errors, max_sample_jump
from .dj import SimDJ
from .plan import Plan, PlanConfig
from .transport import PACKET, Link


@dataclass
class SimConfig:
    handoffs: int = 10
    rtt_a_ms: float = 80.0
    rtt_b_ms: float = 180.0
    jitter_ms: float = 10.0
    loss: float = 0.001
    drift_ppm: float = 50.0        # A läuft +ppm, B −ppm schnell
    seed: int = 1
    human_error_ms: float = 5.0
    mix_bars: int = 8
    hold_bars: int = 2
    arm_bars: int = 6
    block: int = 512
    buffer_s: float = 2.0
    reserve_ms: float = 300.0
    fade_ms: float = 100.0
    fs: int = 48000
    target_buffer_s: float = 3.0   # Zielpuffer des elastischen Ausgangs; 0 = fester Puffer (Scheibe 1)
    min_buffer_s: float = 1.0
    max_dev: float = 0.005
    max_slew: float = 0.001
    deadband_s: float = 0.1
    min_gap_s: float = 60.0        # Mindestabstand zwischen zwei Übergaben
    settle_s: float = 90.0         # Nachlauf nach dem Programmende, damit der Puffer ausregeln kann
    trace_hop_s: float = 0.5       # Abstand der Messpunkte für Puffer/Rate/Zuschauer-Verzögerung


def write_wav(path: str, x: np.ndarray, fs: int) -> None:
    """16-bit-Mono-WAV mit der Standardbibliothek; bei Übersteuerung wird heruntergeregelt."""
    peak = float(np.abs(x).max()) if len(x) else 0.0
    g = 0.89 / peak if peak > 0.89 else 1.0
    pcm = np.round(np.clip(x * g, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())


class OutputMap:
    """Abbildung Ausgabe-Sample ↔ Programmzeit T aus der Aufzeichnung (Position und Rate je Block).

    Der elastische Ausgang liest P(T) mit variabler Rate, sodass Ausgabe-Index und T nicht mehr
    um ein festes D auseinanderliegen. Die Messungen der Simulation laufen in Programmzeit; dafür
    wird der Ausgang über diese Abbildung (exakt, geschlossene Form wie im Mixer) auf das
    T-Raster zurückinterpoliert."""

    def __init__(self, n: int, p0: np.ndarray, r0: np.ndarray, r1: np.ndarray) -> None:
        self.n, self.p0, self.r0, self.r1 = n, p0, r0, r1

    def t_of(self, j: np.ndarray) -> np.ndarray:
        j = np.asarray(j, dtype=np.float64)
        k = np.clip((j // self.n).astype(np.int64), 0, len(self.p0) - 1)
        i = j - k * self.n
        r0, r1 = self.r0[k], self.r1[k]
        return self.p0[k] + r0 * i + (r1 - r0) * i * i / (2.0 * self.n)

    def j_of(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=np.float64)
        k = np.clip(np.searchsorted(self.p0, T, side="right") - 1, 0, len(self.p0) - 1)
        c = T - self.p0[k]
        r0, r1 = self.r0[k], self.r1[k]
        a = (r1 - r0) / (2.0 * self.n)
        i = 2.0 * c / (r0 + np.sqrt(r0 * r0 + 4.0 * a * c))
        return k * self.n + i


class _ProgramView:
    """Programm in Programmzeit (Index = T) als Sicht auf den aufgezeichneten Ausgang."""

    def __init__(self, outs: np.ndarray, omap: OutputMap, length: int) -> None:
        self.outs, self.omap, self.length = outs, omap, length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, sl: slice) -> np.ndarray:
        a, b = sl.start or 0, sl.stop if sl.stop is not None else self.length
        T = np.arange(a, b, dtype=np.float64)
        j = self.omap.j_of(T)
        i0 = max(int(np.floor(j[0])) - 2, 0)
        i1 = int(np.floor(j[-1])) + 4
        # kubisch genügt für die Messungen (Kick-Band 80 Hz, Lücken, Sprünge) und ist 20× billiger
        return frac_read(self.outs[i0:i1], j - i0).astype(np.float32)


def _solve_arrival(stint, t: float) -> float:
    """Wanduhr w, zu der der Programminhalt t im Feed des DJs eintrifft (w − lag(w) = t)."""
    return stint.arrival(t)


def run(cfg: SimConfig, *, keep_program: bool = False, log=None) -> dict:
    t_start = time.time()
    fs = cfg.fs
    plan = Plan(PlanConfig(handoffs=cfg.handoffs, mix_bars=cfg.mix_bars, hold_bars=cfg.hold_bars,
                           arm_bars=cfg.arm_bars, human_error_ms=cfg.human_error_ms,
                           seed=cfg.seed, fs=fs, min_gap_s=cfg.min_gap_s))
    names = plan.cfg.names
    rtt_ms = {names[0]: cfg.rtt_a_ms, names[1]: cfg.rtt_b_ms}
    eps = {names[0]: cfg.drift_ppm * 1e-6, names[1]: -cfg.drift_ppm * 1e-6}
    elastic = cfg.target_buffer_s > 0
    D = int(round(cfg.buffer_s * fs))
    t_stop = int(plan.t_end) + int(cfg.settle_s * fs) if elastic else int(plan.t_end)
    total_t = t_stop + 2 * fs
    # Wanduhr-Obergrenze: Ausgang hinkt um D plus Summe der Lags hinterher (grob: alle RTTs)
    lag_cap = int((cfg.buffer_s + cfg.target_buffer_s + 2.0 + cfg.handoffs
                   * max(cfg.rtt_a_ms, cfg.rtt_b_ms) * 1e-3) * fs) if elastic else D
    w_end = total_t + lag_cap + fs
    rng = np.random.default_rng([cfg.seed, 99])
    djs: dict[str, SimDJ] = {}
    link_stats = {}
    for name in names:
        link = Link(rtt_ms[name] / 2e3, cfg.jitter_ms * 1e-3, cfg.loss)
        n_pk = w_end // PACKET + 4
        up, su = link.loss_mask(n_pk * PACKET, rng)
        down, sd = link.loss_mask(n_pk * PACKET, rng)
        link_stats[name] = {"up": su, "down": sd}
        djs[name] = SimDJ(name, plan, total_t, rtt_s=rtt_ms[name] * 1e-3, eps=eps[name],
                          up_lost=up, down_lost=down)

    mixer = RelayMixer(RelayConfig(
        sample_rate=fs, buffer_s=cfg.buffer_s, reserve_s=cfg.reserve_ms * 1e-3,
        fade_s=cfg.fade_ms * 1e-3, block_size=cfg.block,
        target_buffer_s=cfg.target_buffer_s if elastic else None, min_buffer_s=cfg.min_buffer_s,
        max_dev=cfg.max_dev, max_slew=cfg.max_slew, deadband_s=cfg.deadband_s))
    for name in names:
        mixer.register_source(name, rtt_hint_s=rtt_ms[name] * 1e-3)
    mixer.set_active(names[0])
    if log:
        log(f"Plan: {len(plan.tracks)} Tracks, {plan.t_end / fs:.0f} s Programm, {w_end // cfg.block} Blöcke")

    stage = [0] * len(plan.handoffs)
    snap: list[dict] = [dict() for _ in plan.handoffs]
    outs = np.zeros(w_end, dtype=np.float32)
    aborted = None
    n = cfg.block
    nb = w_end // n
    tr_p0, tr_r0, tr_r1, tr_b = (np.zeros(nb) for _ in range(4))
    k_done = 0
    for w in range(0, w_end - n + 1, n):
        live = mixer.live_edge
        for ho in plan.handoffs:
            st = stage[ho.index]
            if st == 2:
                continue
            if st == 0 and live >= ho.t_arm - fs:
                snap[ho.index]["src_stint"] = djs[ho.src].stints[-1]
                snap[ho.index]["stint"] = djs[ho.dst].start_stint(ho, snap[ho.index]["src_stint"])
                stage[ho.index] = 1
            if stage[ho.index] == 1 and live >= ho.t_h - 2 * fs:
                res = mixer.request_handoff(ho.dst, at=ho.t_h / fs)
                snap[ho.index]["result"] = res
                snap[ho.index]["buffer_request_s"] = mixer.buffer_s
                snap[ho.index]["wall_request"] = w
                snap[ho.index]["params"] = mixer.lag_params(ho.dst)
                snap[ho.index]["info"] = mixer.source_info(ho.dst)
                snap[ho.index]["lag_x"] = mixer.lag_params(ho.src)
                stage[ho.index] = 2
                if not res.accepted:
                    aborted = {"handoff": ho.index, "reason": res.reason, "message": res.message}
                    if log:
                        log(f"Übergabe {ho.index + 1} verweigert: {res.message}")
            break   # immer nur die jeweils nächste Übergabe bearbeiten
        if aborted:
            break
        feeds = {name: djs[name].feed_block(w, n, mixer) for name in names}
        outs[w:w + n] = mixer.process(feeds)
        lb = mixer.last_block
        k = w // n
        if lb is not None:
            tr_p0[k], tr_r0[k], tr_r1[k], tr_b[k] = lb.pos, lb.r0, lb.r1, lb.buffer_s
        else:
            tr_p0[k], tr_r0[k], tr_r1[k], tr_b[k] = w - D, 1.0, 1.0, mixer.buffer_s
        k_done = k + 1
        if mixer.output_time > t_stop:
            break
        if log and (w // n) % 8000 == 0:
            log(f"  {w / fs:6.1f} s / {w_end / fs:.0f} s  aktiv={mixer.status.active}")
    t_sim = time.time() - t_start

    omap = OutputMap(n, tr_p0[:k_done], tr_r0[:k_done], tr_r1[:k_done])
    prog = _ProgramView(outs, omap, int(plan.t_end) + 2 * fs)     # prog[T] = P(T)
    loss_iv = np.array(sorted(i for d in djs.values() for i in d.loss_intervals), dtype=np.float64)
    loss_iv = loss_iv.reshape(-1, 2)

    def P(T0: int, n_: int) -> np.ndarray:
        return prog[T0:T0 + n_]

    handoff_reports = []
    for ho in plan.handoffs:
        sn = snap[ho.index]
        rep: dict = {"index": ho.index + 1, "from": ho.src, "to": ho.dst,
                     "t_handoff_s": ho.t_h / fs}
        res = sn.get("result")
        if res is None:
            rep["accepted"] = False
            rep["reason"] = "nicht erreicht"
            handoff_reports.append(rep)
            continue
        rep["accepted"] = res.accepted
        rep["reason"] = res.reason
        if not res.accepted:
            rep["message"] = res.message
            handoff_reports.append(rep)
            continue
        st = sn["stint"]
        l0, rho, tref = sn["params"]
        xl0, xrho, xtref = sn["lag_x"]
        w_h = _solve_arrival(st, ho.t_h)
        lag_true_abs = float(st.lag(w_h))
        lag_est = l0 + rho * (w_h - tref)
        wl = min(w_h - st.rtt, st.w_cut)
        lx = xl0 + xrho * (wl - xtref)                         # Lag der Programmquelle laut Mixer
        lx_true = st.ref[0] + st.ref[1] * (wl - st.ref[2])      # … und in Wahrheit
        # Der Mixer misst die RTT relativ zu seinem eigenen (geschätzten) Lag der Programmquelle;
        # ein Fehler dort steckt im Raster der Zeitachse und verschiebt nichts gegeneinander.
        lag_true = lx + (lag_true_abs - lx_true)               # wahrer Lag im Raster des Mixers
        rep.update({
            "rtt_true_ms": (lag_true - lx) / fs * 1e3,
            "rtt_est_ms": (lag_est - lx) / fs * 1e3,
            "rtt_error_ms": (lag_est - lag_true) / fs * 1e3,
            "lag_true_ms": lag_true / fs * 1e3,
            "lag_est_ms": lag_est / fs * 1e3,
            "program_lag_before_ms": lx / fs * 1e3,
            "grid_offset_ms": (lx - lx_true) / fs * 1e3,       # aufsummierter Schätzfehler der Achse
            "confidence": float(sn["info"].confidence),
            "drift_est_ppm": float(rho * 1e6),
            "drift_true_ppm": float((djs[ho.dst].eps + xrho) * 1e6),
            "buffer_at_request_s": sn.get("buffer_request_s"),
            "buffer_after_expected_s": res.buffer_after_s,
        })
        # Puffer im Fenster um die Wirkung der Übergabe (gemessen, nicht erwartet)
        kb0 = int(np.clip(w_h // n - int(1.0 * fs) // n, 0, max(k_done - 1, 0)))
        kb1 = int(np.clip(w_h // n + int(8.0 * fs) // n, 1, k_done))
        rep["buffer_min_around_s"] = float(tr_b[kb0:kb1].min()) if kb1 > kb0 else None
        rep["rate_at_handoff"] = float(tr_r0[min(int(w_h // n), max(k_done - 1, 0))])
        # --- Beat-Phase des Programms an der Übergabe
        old, new = ho.old, ho.track
        trk_old, trk_new = plan.track_of(old), plan.track_of(new)
        fade = int(cfg.fade_ms * 1e-3 * fs)

        def ref_old(pos, old=old):
            return frac_read(trk_old.low, pos - old.start).astype(np.float64)

        def ref_new(pos, new=new):
            return frac_read(trk_new.low, pos - (new.start - new.human_s * fs)).astype(np.float64)

        swap = plan.swap_point(new)
        kt = trk_old.kick_times + old.start
        post_k = kt[(kt > ho.t_h + fade + 0.25 * fs) & (kt < swap - 0.15 * fs)][:10]
        pre_k = kt[(kt > ho.t_h - 2.5 * fs) & (kt < ho.t_h - 0.15 * fs)][-4:]
        pad = int(0.35 * fs)

        def errs(kicks, ref):
            ks = [k for k in kicks if 0 <= k - pad and k + pad < len(prog)]
            return kick_phase_errors(prog, ref, np.array(ks), fs, loss_iv)

        pre = errs(pre_k, ref_old)
        post = errs(post_k, ref_old)
        pre_v = [r["err_ms"] for r in pre if r["ok"]]
        post_v = [r["err_ms"] for r in post if r["ok"]]
        pre_off = float(np.median(pre_v)) if pre_v else 0.0
        if post_v:
            rel = [v - pre_off for v in post_v]
            rep["beat_phase_error_ms"] = float(np.median(rel))
            rep["beat_phase_error_max_abs_ms"] = float(np.max(np.abs(rel)))
        else:
            rep["beat_phase_error_ms"] = rep["beat_phase_error_max_abs_ms"] = None
        rep["beat_kicks"] = {"before": pre, "after": post}
        rep["beat_offset_before_ms"] = pre_off if pre_v else None
        rep["beat_kicks_measured"] = {"before": len(pre_v), "after": len(post_v),
                                      "skipped_loss": sum(1 for r in pre + post if r.get("lost"))}
        # neuer Track nach dem Bass-Swap: menschlicher Fehler + Relay-Anteil
        kn = trk_new.kick_times + new.start
        new_k = kn[(kn > swap + 0.15 * fs) & (kn < swap + 4 * plan.bar)][:10]
        newr = errs(new_k, ref_new)
        new_v = [r["err_ms"] for r in newr if r["ok"]]
        rep["beat_kicks"]["new_track"] = newr
        rep["dj_human_error_ms"] = new.human_s * 1e3
        if new_v:
            rep["new_track_phase_error_ms"] = float(np.median(new_v))
            rep["new_track_relay_error_ms"] = float(np.median(new_v)) - new.human_s * 1e3
        else:
            rep["new_track_phase_error_ms"] = rep["new_track_relay_error_ms"] = None
        # --- Sprünge und Lücken rund um die Übergabe
        a, b = int(ho.t_h - 0.1 * fs), int(ho.t_h + fade + 0.1 * fs)
        seg = P(a, b - a)
        base = P(int(ho.t_h - 1.5 * fs), int(1.4 * fs))

        def loss_mask(T0, nn):
            m = np.zeros(nn, dtype=bool)
            for lo, hi in loss_iv[(loss_iv[:, 1] > T0 - 8) & (loss_iv[:, 0] < T0 + nn + 8)]:
                m[max(int(lo) - 8 - T0, 0):max(int(hi) + 8 - T0, 0)] = True
            return m

        rep["max_sample_jump"] = max_sample_jump(seg, loss_mask(a, b - a))
        rep["max_sample_jump_baseline"] = max_sample_jump(base, loss_mask(int(ho.t_h - 1.5 * fs), len(base)))
        ga = int(ho.t_h - 1.0 * fs)
        gaps = find_gaps(P(ga, int(4.0 * fs)), fs)
        glist = []
        for s0, ln in gaps:
            T0 = ga + s0
            expl = bool(len(loss_iv) and np.any((loss_iv[:, 1] > T0 - 0.002 * fs)
                                                & (loss_iv[:, 0] < T0 + ln + 0.002 * fs)))
            glist.append({"start_s": T0 / fs, "duration_ms": ln / fs * 1e3, "explained_by_loss": expl})
        rep["gaps"] = glist
        rep["gaps_unexplained"] = sum(1 for g in glist if not g["explained_by_loss"])
        rep["max_gap_unexplained_ms"] = max([g["duration_ms"] for g in glist
                                             if not g["explained_by_loss"]], default=0.0)
        handoff_reports.append(rep)

    # globale Lückenprüfung über das ganze Programm
    # Speicher freigeben, bevor die langen Auswertungen laufen (DJ-Master, gerenderte Tracks)
    rtt_of = {nm: dj.rtt for nm, dj in djs.items()}
    for dj in djs.values():
        dj.M = None
    plan._cache.clear()
    # Lücken im Ausgang (Ausgabe-Raster), danach in Programmzeit umgerechnet
    j_lo = int(np.ceil(omap.j_of(np.array([0.0]))[0]))
    j_hi = min(int(omap.j_of(np.array([float(plan.t_end)]))[0]), n * k_done)
    all_gaps = []
    for g0, gl in find_gaps_chunked(outs[j_lo:j_hi], fs):
        t0 = float(omap.t_of(np.array([j_lo + g0]))[0])
        t1 = float(omap.t_of(np.array([j_lo + g0 + gl]))[0])
        all_gaps.append((t0, t1 - t0))
    unexpl = 0
    for s0, ln in all_gaps:
        if not (len(loss_iv) and np.any((loss_iv[:, 1] > s0 - 0.002 * fs)
                                        & (loss_iv[:, 0] < s0 + ln + 0.002 * fs))):
            unexpl += 1
    ok_h = [r for r in handoff_reports if r.get("accepted")]

    # --- Puffer, Rate und Zuschauer-Verzögerung über die Sitzung
    kk = np.arange(k_done)
    p_end = tr_p0[:k_done] + n * 0.5 * (tr_r0[:k_done] + tr_r1[:k_done])
    rate_blk = 0.5 * (tr_r0[:k_done] + tr_r1[:k_done])
    live = [(ho.t_h, snap[ho.index]["stint"], names.index(ho.dst)) for ho in plan.handoffs
            if snap[ho.index].get("result") is not None and snap[ho.index]["result"].accepted]
    first_stint = djs[names[0]].stints[0]
    series = []
    delays = []
    hop = max(1, int(cfg.trace_hop_s * fs / n))
    series_hop = max(1, int(10.0 * fs / n))
    for k in range(0, k_done, hop):
        T_out = float(p_end[k])
        if T_out < 0:
            continue
        cur, who = first_stint, 0
        for t_h, st_h, wi in live:
            if t_h <= T_out:
                cur, who = st_h, wi
        w_arr = _solve_arrival(cur, T_out)
        w_play = w_arr - rtt_of[names[who]] / 2.0
        d_s = ((k + 1) * n - w_play) / fs
        delays.append((k, d_s))
        if k % series_hop < hop:
            series.append({"t_s": round((k + 1) * n / fs, 1), "buffer_s": round(float(tr_b[k]), 3),
                           "rate": round(float(rate_blk[k]), 6),
                           "bpm_equiv": round(174.0 * float(rate_blk[k]), 2),
                           "viewer_delay_s": round(d_s, 3), "dj": names[who]})
    d_arr = np.array([d for _, d in delays]) if delays else np.zeros(1)
    live_k = kk[p_end >= 0]
    b_live = tr_b[:k_done][live_k] if len(live_k) else np.zeros(1)
    r_live = rate_blk[live_k] if len(live_k) else np.ones(1)
    buffer_rep = {
        "target_s": cfg.target_buffer_s if elastic else None,
        "min_limit_s": cfg.min_buffer_s if elastic else None,
        "min_s": float(b_live.min()), "max_s": float(b_live.max()), "final_s": float(b_live[-1]),
        "start_s": float(b_live[0]),
        "output_delay_final_s": float((k_done * n - p_end[-1]) / fs),
    }
    rate_rep = {
        "min": float(r_live.min()), "max": float(r_live.max()), "final": float(r_live[-1]),
        "max_abs_dev": float(np.abs(r_live - 1.0).max()),
        "bpm_min": 174.0 * float(r_live.min()), "bpm_max": 174.0 * float(r_live.max()),
        "bpm_final": 174.0 * float(r_live[-1]),
    }
    viewer_rep = {
        "definition": "Wanduhr − Zeitpunkt, zu dem der jeweils live spielende DJ den gerade "
                      "ausgegebenen Inhalt gespielt hat (Ankunft beim Server − RTT/2)",
        "max_s": float(d_arr.max()), "min_s": float(d_arr.min()), "mean_s": float(d_arr.mean()),
        "final_s": float(d_arr[-1]),
    }

    def worst(key):
        v = [abs(r[key]) for r in ok_h if r.get(key) is not None]
        return max(v) if v else None

    summary = {
        "handoffs_planned": cfg.handoffs,
        "handoffs_accepted": len(ok_h),
        "max_abs_rtt_error_ms": worst("rtt_error_ms"),
        "max_abs_beat_phase_error_ms": worst("beat_phase_error_max_abs_ms"),
        "max_abs_new_track_relay_error_ms": worst("new_track_relay_error_ms"),
        "max_dj_human_error_ms": worst("dj_human_error_ms"),
        "gaps_total": len(all_gaps),
        "gaps_unexplained": unexpl,
        "max_program_lag_ms": max([r["lag_est_ms"] for r in ok_h], default=None),
        "buffer_min_s": buffer_rep["min_s"], "buffer_max_s": buffer_rep["max_s"],
        "buffer_final_s": buffer_rep["final_s"],
        "rate_min": rate_rep["min"], "rate_max": rate_rep["max"],
        "viewer_delay_max_s": viewer_rep["max_s"], "viewer_delay_final_s": viewer_rep["final_s"],
        "runtime_s": round(time.time() - t_start, 1),
        "simulation_s": round(t_sim, 1),
        "program_s": plan.t_end / fs,
    }
    acc = {
        "all_handoffs_accepted": len(ok_h) == cfg.handoffs and aborted is None,
        "rtt_error_below_1ms": summary["max_abs_rtt_error_ms"] is not None
                               and summary["max_abs_rtt_error_ms"] < 1.0,
        "beat_phase_below_3ms": summary["max_abs_beat_phase_error_ms"] is not None
                                and summary["max_abs_beat_phase_error_ms"] < 3.0,
        "no_unexplained_gap_over_5ms": unexpl == 0,
    }
    if elastic:
        acc.update({
            "buffer_never_below_min": buffer_rep["min_s"] >= cfg.min_buffer_s,
            "viewer_delay_below_4s": viewer_rep["max_s"] < 4.0,
            "rate_within_max_dev": rate_rep["max_abs_dev"] <= cfg.max_dev + 1e-9,
            "buffer_ends_near_target": abs(buffer_rep["final_s"] - cfg.target_buffer_s) <= cfg.deadband_s,
        })
    acc["passed"] = all(acc.values())
    report = {
        "config": asdict(cfg),
        "summary": summary,
        "acceptance": acc,
        "buffer": buffer_rep,
        "rate": rate_rep,
        "viewer_delay": viewer_rep,
        "series": series,
        "link_stats": link_stats,
        "handoffs": handoff_reports,
        "mixer_events": [{"wall_s": e.wall / fs, "kind": e.kind, **{k: (v if not isinstance(v, np.generic) else v.item())
                                                                   for k, v in e.detail.items()}}
                         for e in mixer.events],
    }
    if aborted:
        report["aborted"] = aborted
    if keep_program:
        report["_program"] = outs[:j_hi]       # Ausgang, wie er gehört würde (mit Tempo-Regelung)
    return report
