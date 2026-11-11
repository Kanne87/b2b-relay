"""Auswertung des Programm-Mitschnitts (WAV + Trace des Servers) je Netzprofil.

Eingaben: ``program-*.wav`` und ``program-*.trace.jsonl`` (vom Server geschrieben, ``B2B_RECORD_DIR``),
optional ``status.jsonl`` (Orchestrator, GET /status alle 2 s).

Messgrößen je Übergabe (Schalt-Ereignis ``handoff_effective`` im Trace):

* ``phase_jump_ms``: Beat-Phasenfehler. Kick-Onsets (Schwellendurchgang) in 22 s vor und 18 s nach der
  Übergabe (Fenster 2 s vor bis 8 s nach dem Umschaltpunkt ausgelassen: Überblendung und Ausblenden der
  durchgereichten Kicks des Abgebenden), gemessen in **Inhaltszeit** ``Tc`` (Programmzeit ohne Umlabeln,
  unabhängig vom Varispeed-Tempo), Rasterphase als Kreismittel modulo Beat (174 BPM). Sprung = Differenz
  der Phasen nach/vor, auf ±½ Beat gefaltet.
* ``null_jump_ms`` (je Haltezeit, ohne Übergabe): dieselbe Messung an einem Punkt mitten in der Haltezeit, an
  dem nichts umschaltet - das Grundrauschen der Messung unter der Netzstörung (JackTrip verschiebt den Feed
  der live spielenden Quelle bei Unter-/Überlauf und Verlust um ganze Slots, 5,3 ms). Ein Phasensprung an
  einer Übergabe ist nur dann ein Fehler des Relays, wenn er deutlich über diesem Rauschen liegt.
* ``gaps``: Läufe digitaler Stille ≥ 5 ms im Fenster [−1 s, +3 s]; ``max_jump_ratio``: größter Sample-Sprung
  im Fenster [−0,15 s, +0,35 s] gegen das Vergleichsfenster [−3 s, −2 s] (Klicks).
* gesamt: xruns, Lücken im ganzen Mitschnitt, Puffer- und Raten-Verlauf, Drift; ``concealed``: Aussetzer der
  Strecke (exakte Nullen im Feed einer Quelle, z. B. Paketverlust), die der Server verschleiert hat (Trace-Feld
  ``cc``) - sie sind keine Lücken im Programm, zeigen aber, wie oft die Strecke ausgesetzt hat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import signals as sg  # noqa: E402

from b2b_relay.live.wav import read_wav  # noqa: E402
from b2b_relay.sim.analysis import find_gaps_chunked, max_sample_jump  # noqa: E402

FS = 48000

LIMITS = {   # Abnahmegrenzen je Profil (ms bzw. Anzahl); Lücken: Aussetzer der Strecke verschleiert der Server
    "clean": dict(phase_ms=1.0, unexplained_gaps=0),
    "wan40": dict(phase_ms=1.0, unexplained_gaps=0),
    "wan120": dict(phase_ms=10.0, unexplained_gaps=0),
}


def load_trace(path: Path) -> tuple[list[dict], list[dict]]:
    blk, ev = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        (blk if r["k"] == "blk" else ev).append(r)
    return blk, ev


class TimeMap:
    """Mitschnitt-Index ↔ Inhaltszeit Tc (Programmzeit ohne Umlabeln) aus den Block-Zeilen."""

    def __init__(self, blk: list[dict]) -> None:
        self.i = np.array([b["i"] for b in blk], dtype=np.float64)
        self.t = np.array([b["Tc"] for b in blk], dtype=np.float64)
        order = np.argsort(self.t)
        self.ts = self.t[order]
        self.is_ = self.i[order]

    def to_t(self, idx):
        return np.interp(idx, self.i, self.t)

    def to_idx(self, tc):
        return np.interp(tc, self.ts, self.is_)


PRE = int(0.12 * FS)        # Vorlauf der Onset-Erkennung vor dem Fenster (> Ruhephase ``refractory`` 0,08 s)


def _onsets_tc(mid: np.ndarray, tm: TimeMap, lo: int, hi: int) -> np.ndarray:
    """Onsets in [lo, hi) als Inhaltszeit. Erkannt wird ab ``lo − PRE``: beginnt das Fenster mitten in einem
    Kick oder seinem Ausklang, wäre der erste Schwellendurchgang sonst ein falscher Onset ohne Ruhephase davor
    (eine Fehlmessung unter 64 verschob das Kreismittel um bis zu 0,8 ms, S3904)."""
    lo, hi = max(lo, 0), min(hi, len(mid))
    if hi - lo < FS:
        return np.empty(0)
    a = max(lo - PRE, 0)
    ons = sg.detect_onsets(mid[a:hi]) + a
    ons = ons[ons >= max(lo, a + PRE)]
    return tm.to_t(ons)


WINDOWS = ((-24.0, -2.0), (8.0, 26.0))     # Fenster vor/nach dem Umschaltpunkt (s) für den Phasensprung


def phase_jump(mid: np.ndarray, tm: TimeMap, i_sw: int, windows=WINDOWS) -> dict:
    (b0, b1), (a0, a1) = windows
    before = _onsets_tc(mid, tm, i_sw + int(b0 * FS), i_sw + int(b1 * FS))
    after = _onsets_tc(mid, tm, i_sw + int(a0 * FS), i_sw + int(a1 * FS))
    out = {"n_before": int(len(before)), "n_after": int(len(after)), "jump_ms": None}
    if len(before) < 8 or len(after) < 8:
        out["note"] = "zu wenige Kicks in einem Fenster"
        return out
    pb, rb = sg.grid_phase(before)
    pa, ra = sg.grid_phase(after)
    out.update(jump_ms=sg.wrap(pa - pb) / FS * 1e3, r_before=round(rb, 3), r_after=round(ra, 3))
    return out


def null_jumps(mid: np.ndarray, tm: TimeMap, switches: list[int], windows=WINDOWS) -> list[dict]:
    """Phasensprung-Messung ohne Übergabe: je Abschnitt zwischen zwei Umschaltpunkten (und vor dem ersten) ein
    Punkt, dessen Fenster ganz im Abschnitt liegen (8 s nach dem einen, 2 s vor dem nächsten Umschaltpunkt)."""
    (b0, b1), (a0, a1) = windows
    bounds = [int(8 * FS)] + [i + int(8 * FS) for i in switches]
    ends = [i - int(2 * FS) for i in switches] + [len(mid) - int(2 * FS)]
    out = []
    for lo, hi in zip(bounds, ends):
        c_min, c_max = lo - int(b0 * FS), hi - int(a1 * FS)
        if c_max < c_min:
            continue
        c = (c_min + c_max) // 2
        pj = phase_jump(mid, tm, c, windows)
        if pj["jump_ms"] is not None:
            out.append({"at_s": round(c / FS, 2), "jump_ms": round(pj["jump_ms"], 3)})
    return out


def analyze_profile(wav: Path, trace: Path, name: str, status: Path | None = None,
                    handovers: list[dict] | None = None, windows=WINDOWS) -> dict:
    fs, stereo = read_wav(wav)
    assert fs == FS, fs
    mid = ((stereo[:, 0] + stereo[:, 1]) * 0.5).astype(np.float32)
    blk, ev = load_trace(trace)
    tm = TimeMap(blk)
    eff = [e for e in ev if e["kind"] == "handoff_effective"]
    falls = [e for e in ev if e["kind"] == "fallback"]
    refused = [e for e in ev if e["kind"] == "handoff_refused"]
    sched = [e for e in ev if e["kind"] == "handoff_scheduled"]
    per = []
    for n, e in enumerate(eff, 1):
        i_sw = int(tm.to_idx(e["Tc_switch"]))
        pj = phase_jump(mid, tm, i_sw, windows)
        lag = next((x.get("lag_s") for x in sched if x.get("t_switch") == e.get("t_switch")
                    and x.get("target") == e.get("source")), None)
        w = mid[max(i_sw - FS, 0):i_sw + 3 * FS]
        gaps = find_gaps_chunked(w, FS, 5.0)
        j_win = max_sample_jump(mid[max(i_sw - int(0.15 * FS), 0):i_sw + int(0.35 * FS)])
        j_base = max_sample_jump(mid[max(i_sw - 3 * FS, 0):i_sw - 2 * FS]) or 1e-9
        per.append({"n": n, "to": e.get("source"), "from": e.get("previous"), "switch_s": round(i_sw / FS, 2),
                    "phase_jump_ms": None if pj["jump_ms"] is None else round(pj["jump_ms"], 3),
                    "phase_detail": pj, "lag_s": lag,
                    "gaps": len(gaps), "max_gap_ms": round(max((g[1] for g in gaps), default=0) / FS * 1e3, 2),
                    "max_jump_ratio": round(j_win / j_base, 2)})
    nulls = null_jumps(mid, tm, [int(tm.to_idx(e["Tc_switch"])) for e in eff], windows)
    gaps_all = find_gaps_chunked(mid[int(8 * FS):], FS, 5.0)
    b = np.array([x["b"] for x in blk])
    r = np.array([x["r0"] for x in blk])
    d = np.array([x["drift_ms"] for x in blk])
    step = max(1, int(5 * FS / max(1, int(np.median([x["n"] for x in blk])))))
    series = [{"t_s": round(x["i"] / FS, 1), "buffer_s": x["b"], "rate": x["r0"], "drift_ms": x["drift_ms"]}
              for x in blk[::step]]
    jumps = [p["phase_jump_ms"] for p in per if p["phase_jump_ms"] is not None]
    res = {
        "profile": name, "wav": str(wav), "duration_s": round(len(mid) / FS, 1),
        "handovers_effective": len(eff), "handovers_refused": len(refused), "fallbacks": len(falls),
        "per_handover": per,
        "null_jumps": nulls,
        "summary": {
            "phase_jump_max_abs_ms": round(max((abs(j) for j in jumps), default=float("nan")), 3),
            "phase_jump_mean_abs_ms": round(float(np.mean(np.abs(jumps))), 3) if jumps else None,
            "phase_jump_missing": len(per) - len(jumps),
            "null_jump_max_abs_ms": round(max((abs(x["jump_ms"]) for x in nulls), default=float("nan")), 3),
            "gaps_total": len(gaps_all), "max_gap_ms": round(max((g[1] for g in gaps_all), default=0) / FS * 1e3, 2),
            "max_jump_ratio": max((p["max_jump_ratio"] for p in per), default=None),
            "xruns": int(blk[-1]["xr"]) if blk else None,
            # nur während des Mitschnitts (die Startphase von jackd/b2b-relay liegt davor und trifft kein Programm)
            "xruns_recording": int(blk[-1]["xr"] - blk[0]["xr"]) if blk else None,
            "buffer_min_s": round(float(b.min()), 3), "buffer_max_s": round(float(b.max()), 3),
            "rate_min": round(float(r.min()), 5), "rate_max": round(float(r.max()), 5),
            "drift_max_abs_ms": round(float(np.abs(d[len(d) // 10:]).max()), 3) if len(d) else None,
            "drift_final_ms": round(float(d[-1]), 3) if len(d) else None,
            "concealed": int(blk[-1].get("cc", 0) - blk[0].get("cc", 0)) if blk else None,
        },
        "series": series,
    }
    if status and Path(status).exists():
        last = None
        for line in Path(status).read_text().splitlines():
            if line.strip():
                last = json.loads(line)
        if last:
            res["status_final"] = {k: last.get(k) for k in ("live", "jack", "worker", "underruns", "drift", "buffer")}
    if handovers is not None:
        res["handover_requests"] = handovers
    res["acceptance"] = acceptance(res, name)
    return res


def acceptance(res: dict, profile: str) -> dict:
    lim = LIMITS.get(profile, LIMITS["wan120"])
    s = res["summary"]
    n_req = len(res.get("handover_requests", [])) or res["handovers_effective"]
    ok = {
        "all_handovers_effective": res["handovers_effective"] >= n_req and n_req > 0,
        "no_phase_jump_missing": s["phase_jump_missing"] == 0,
        f"phase_jump<={lim['phase_ms']}ms": bool(s["phase_jump_max_abs_ms"] <= lim["phase_ms"]),
        "no_fallback": res["fallbacks"] == 0,
        "xruns==0": s["xruns_recording"] == 0,
        "buffer>=min": s["buffer_min_s"] >= 0.95,
        "rate_within_0.5%": s["rate_min"] >= 0.995 - 1e-4 and s["rate_max"] <= 1.005 + 1e-4,
        "drift<10ms": s["drift_max_abs_ms"] is not None and s["drift_max_abs_ms"] < 10.0,
    }
    if lim["unexplained_gaps"] is not None:
        ok["no_gaps"] = s["gaps_total"] <= lim["unexplained_gaps"]
    ok["passed"] = all(ok.values())
    return ok


def incomplete(profile: str, error: str, handovers: list[dict] | None = None) -> dict:
    """Ergebnis eines Profils ohne auswertbaren Mitschnitt (Abbruch vor dem Start der Aufnahme)."""
    res = {"profile": profile, "error": error, "completed": False, "handovers_effective": 0,
           "handovers_refused": 0, "fallbacks": 0, "per_handover": [], "summary": {}, "series": []}
    if handovers is not None:
        res["handover_requests"] = handovers
    res["acceptance"] = {"completed": False, "passed": False}
    return res


def mark_incomplete(res: dict, error: str) -> dict:
    """Ausgewertetes, aber abgebrochenes Profil: Fehler vermerken, Abnahme gilt als verfehlt."""
    res = dict(res, error=error, completed=False)
    res["acceptance"] = dict(res["acceptance"], completed=False, passed=False)
    return res


def markdown(results: list[dict]) -> str:
    out = ["# b2b-relay Integrationstest (Scheibe 2a)", ""]
    out += ["| Profil | Übergaben | Sprung max (ms) | Sprung Mittel (ms) | ohne Übergabe max (ms) | Lücken | max Lücke (ms) | verschleiert | xruns "
            "| Puffer min/max (s) | Rate min/max | Drift max (ms) | Ergebnis |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r["summary"]
        if not s:
            out.append(f"| {r['profile']} | – | – | – | – | – | – | – | – | – | – | – | abgebrochen |")
            continue
        out.append(f"| {r['profile']} | {r['handovers_effective']} | {s['phase_jump_max_abs_ms']} | "
                   f"{s['phase_jump_mean_abs_ms']} | {s.get('null_jump_max_abs_ms')} | {s['gaps_total']} | {s['max_gap_ms']} | {s.get('concealed')} | "
                   f"{s['xruns']} | "
                   f"{s['buffer_min_s']}/{s['buffer_max_s']} | {s['rate_min']}/{s['rate_max']} | "
                   f"{s['drift_max_abs_ms']} | {'bestanden' if r['acceptance']['passed'] else 'NICHT bestanden'} |")
    queues = sorted({str(r["jacktrip_queue"]) for r in results if r.get("jacktrip_queue")})
    if queues:
        out += ["", f"JackTrip-Queue des Hubs: {', '.join(queues)} (Slots à 256 Frames)"]
    for r in results:
        out += ["", f"## {r['profile']}", ""]
        if r.get("error"):
            out += [f"**Abgebrochen:** {r['error']} (Container-Logs: `{r['profile']}/logs/`, `compose.log`)", ""]
        out += ["| # | von → nach | Umschaltpunkt (s) | Phasensprung (ms) | Lücken | "
                "max Sprung-Verhältnis |", "|---|---|---|---|---|---|"]
        for p in r["per_handover"]:
            out.append(f"| {p['n']} | {p['from']} → {p['to']} | {p['switch_s']} | {p['phase_jump_ms']} | "
                       f"{p['gaps']} | {p['max_jump_ratio']} |")
        failed = [k for k, v in r["acceptance"].items() if not v]
        out += ["", "Abnahme: " + ("alle Kriterien erfüllt." if not failed else "verfehlt: " + ", ".join(failed))]
    out += ["", "Phasensprung = Differenz der Beat-Rasterphase (174 BPM, Inhaltszeit) 22 s vor und 18 s nach dem "
            "Umschaltpunkt; „ohne Übergabe“ = dieselbe Messung mitten in einer Haltezeit (Grundrauschen); "
            "„verschleiert“ = Aussetzer der Strecke, die der Server überbrückt hat; Netzstörung (tc netem) wirkt "
            "nur auf den Upload der Clients (siehe README)."]
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wav", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--profile", default="clean")
    ap.add_argument("--status")
    ap.add_argument("--out", default="analysis.json")
    a = ap.parse_args(argv)
    res = analyze_profile(Path(a.wav), Path(a.trace), a.profile, Path(a.status) if a.status else None)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(markdown([res]))
    return 0 if res["acceptance"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
