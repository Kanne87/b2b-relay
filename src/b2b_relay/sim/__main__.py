"""CLI: python -m b2b_relay.sim --handoffs 10 --rtt-a-ms 80 --rtt-b-ms 180 --out report.json"""

from __future__ import annotations

import argparse
import json
import sys

from .scenario import SimConfig, run, write_wav


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m b2b_relay.sim",
        description="Deterministische Offline-Simulation des Relay-Mixers (A→B→A…).")
    ap.add_argument("--handoffs", type=int, default=10, help="Anzahl Übergaben (Standard 10)")
    ap.add_argument("--rtt-a-ms", type=float, default=80.0, help="RTT von DJ A in ms")
    ap.add_argument("--rtt-b-ms", type=float, default=180.0, help="RTT von DJ B in ms")
    ap.add_argument("--jitter-ms", type=float, default=10.0, help="Jitter-Puffer je Richtung in ms")
    ap.add_argument("--loss", type=float, default=0.001, help="Paketverlust je Richtung (0..1)")
    ap.add_argument("--drift-ppm", type=float, default=50.0,
                    help="Taktdrift: A läuft +ppm, B −ppm schneller")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--human-error-ms", type=float, default=5.0,
                    help="menschlicher Fehler beim Einstarten (±, gleichverteilt)")
    ap.add_argument("--mix-bars", type=int, default=8, help="Länge einer Überblendung in Takten")
    ap.add_argument("--buffer-s", type=float, default=2.0, help="Ausgabepuffer D in Sekunden")
    ap.add_argument("--reserve-ms", type=float, default=300.0)
    ap.add_argument("--fade-ms", type=float, default=100.0, help="Crossfade der Übergabe (50–200)")
    ap.add_argument("--block", type=int, default=512, help="Blockgröße in Samples")
    ap.add_argument("--out", help="Report als JSON")
    ap.add_argument("--wav", help="Programm als 16-bit-WAV (Mono, 48 kHz)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    cfg = SimConfig(handoffs=a.handoffs, rtt_a_ms=a.rtt_a_ms, rtt_b_ms=a.rtt_b_ms,
                    jitter_ms=a.jitter_ms, loss=a.loss, drift_ppm=a.drift_ppm, seed=a.seed,
                    human_error_ms=a.human_error_ms, mix_bars=a.mix_bars, buffer_s=a.buffer_s,
                    reserve_ms=a.reserve_ms, fade_ms=a.fade_ms, block=a.block)
    log = None if a.quiet else (lambda m: print(m, file=sys.stderr, flush=True))
    rep = run(cfg, keep_program=bool(a.wav), log=log)
    prog = rep.pop("_program", None)
    if a.wav and prog is not None:
        write_wav(a.wav, prog, cfg.fs)
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    s, acc = rep["summary"], rep["acceptance"]
    print(f"Übergaben {s['handoffs_accepted']}/{s['handoffs_planned']}  "
          f"RTT-Fehler max {s['max_abs_rtt_error_ms']} ms  "
          f"Beat-Phase max {s['max_abs_beat_phase_error_ms']} ms  "
          f"Lücken unerklärt {s['gaps_unexplained']}  Laufzeit {s['runtime_s']} s  "
          f"-> {'BESTANDEN' if acc['passed'] else 'NICHT BESTANDEN'}")
    return 0 if acc["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
