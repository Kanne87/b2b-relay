"""Szenario A→B→A… : Simulation durchlaufen lassen und je Übergabe vermessen."""

from __future__ import annotations

import time
import wave
from dataclasses import asdict, dataclass

import numpy as np

from ..mixer import RelayConfig, RelayMixer
from ..timeline import frac_read
from .analysis import find_gaps, kick_phase_errors, max_sample_jump
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


def _solve_arrival(stint, t: float) -> float:
    """Wanduhr w, zu der der Programminhalt t im Feed des DJs eintrifft (w − lag(w) = t)."""
    w = t + float(stint.lag(t + 1.0))
    for _ in range(6):
        w = t + float(stint.lag(w))
    return w


def run(cfg: SimConfig, *, keep_program: bool = False, log=None) -> dict:
    t_start = time.time()
    fs = cfg.fs
    plan = Plan(PlanConfig(handoffs=cfg.handoffs, mix_bars=cfg.mix_bars, hold_bars=cfg.hold_bars,
                           arm_bars=cfg.arm_bars, human_error_ms=cfg.human_error_ms,
                           seed=cfg.seed, fs=fs))
    names = plan.cfg.names
    rtt_ms = {names[0]: cfg.rtt_a_ms, names[1]: cfg.rtt_b_ms}
    eps = {names[0]: cfg.drift_ppm * 1e-6, names[1]: -cfg.drift_ppm * 1e-6}
    D = int(round(cfg.buffer_s * fs))
    total_t = int(plan.t_end) + 2 * fs
    w_end = total_t + D + fs
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

    mixer = RelayMixer(RelayConfig(sample_rate=fs, buffer_s=cfg.buffer_s,
                                   reserve_s=cfg.reserve_ms * 1e-3, fade_s=cfg.fade_ms * 1e-3))
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
    for w in range(0, w_end - n + 1, n):
        live = mixer.live_edge
        for ho in plan.handoffs:
            st = stage[ho.index]
            if st == 2:
                continue
            if st == 0 and live >= ho.t_arm - fs:
                snap[ho.index]["stint"] = djs[ho.dst].start_stint(ho, mixer)
                stage[ho.index] = 1
            if stage[ho.index] == 1 and live >= ho.t_h - 2 * fs:
                res = mixer.request_handoff(ho.dst, at=ho.t_h / fs)
                snap[ho.index]["result"] = res
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
        if log and (w // n) % 8000 == 0:
            log(f"  {w / fs:6.1f} s / {w_end / fs:.0f} s  aktiv={mixer.status.active}")
    t_sim = time.time() - t_start

    prog = outs      # prog[T + D] = P(T)
    loss_iv = np.array(sorted(i for d in djs.values() for i in d.loss_intervals), dtype=np.float64)
    loss_iv = loss_iv.reshape(-1, 2)

    def P(T0: int, n_: int) -> np.ndarray:
        return prog[T0 + D:T0 + D + n_]

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
        lag_true = float(st.lag(w_h))
        lag_est = l0 + rho * (w_h - tref)
        wl = min(w_h - st.rtt, st.w_cut)
        lx = xl0 + xrho * (wl - xtref)
        rep.update({
            "rtt_true_ms": (lag_true - lx) / fs * 1e3,
            "rtt_est_ms": (lag_est - lx) / fs * 1e3,
            "rtt_error_ms": (lag_est - lag_true) / fs * 1e3,
            "lag_true_ms": lag_true / fs * 1e3,
            "lag_est_ms": lag_est / fs * 1e3,
            "program_lag_before_ms": lx / fs * 1e3,
            "confidence": float(sn["info"].confidence),
            "drift_est_ppm": float(rho * 1e6),
            "drift_true_ppm": float((djs[ho.dst].eps + xrho) * 1e6),
        })
        # --- Beat-Phase des Programms an der Übergabe
        old, new = ho.old, ho.track
        fade = int(cfg.fade_ms * 1e-3 * fs)

        def ref_old(pos, old=old):
            return frac_read(old.track.low, pos - old.start).astype(np.float64)

        def ref_new(pos, new=new):
            return frac_read(new.track.low, pos - (new.start - new.human_s * fs)).astype(np.float64)

        swap = plan.swap_point(new)
        kt = old.track.kick_times + old.start
        post_k = kt[(kt > ho.t_h + fade + 0.25 * fs) & (kt < swap - 0.15 * fs)][:6]
        pre_k = kt[(kt > ho.t_h - 2.5 * fs) & (kt < ho.t_h - 0.15 * fs)][-4:]
        pad = int(0.35 * fs)

        def errs(kicks, ref):
            ks = [k for k in kicks if 0 <= k + D - pad and k + D + pad < len(prog)]
            return kick_phase_errors(prog[D:], ref, np.array(ks), fs, loss_iv)

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
        kn = new.track.kick_times + new.start
        new_k = kn[(kn > swap + 0.15 * fs) & (kn < swap + 4 * plan.bar)][:6]
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
    all_gaps = find_gaps(prog[D:D + int(plan.t_end)], fs)
    unexpl = 0
    for s0, ln in all_gaps:
        if not (len(loss_iv) and np.any((loss_iv[:, 1] > s0 - 0.002 * fs)
                                        & (loss_iv[:, 0] < s0 + ln + 0.002 * fs))):
            unexpl += 1
    ok_h = [r for r in handoff_reports if r.get("accepted")]

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
    acc["passed"] = all(acc.values())
    report = {
        "config": asdict(cfg),
        "summary": summary,
        "acceptance": acc,
        "link_stats": link_stats,
        "handoffs": handoff_reports,
        "mixer_events": [{"wall_s": e.wall / fs, "kind": e.kind, **{k: (v if not isinstance(v, np.generic) else v.item())
                                                                   for k, v in e.detail.items()}}
                         for e in mixer.events],
    }
    if aborted:
        report["aborted"] = aborted
    if keep_program:
        report["_program"] = prog[D:D + int(plan.t_end)]
    return report
