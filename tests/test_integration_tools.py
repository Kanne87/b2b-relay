"""Werkzeuge des Integrationstests (Signale, Auswertung, Ablaufplan) ohne Docker."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ITEST = Path(__file__).resolve().parent / "integration"
sys.path.insert(0, str(ITEST))

import analyze  # noqa: E402
import signals as sg  # noqa: E402
from b2b_relay.live.wav import WavWriter  # noqa: E402

FS = 48000


def test_onsets_are_precise_and_timbres_share_the_same_onset():
    ph = 1234.5
    for name in "ABC":
        x = sg.render_track(name, 8, ph)
        ons = sg.detect_onsets(x)
        assert len(ons) >= 22
        err = (ons - ph) % sg.BEAT
        assert 7 < np.median(err) < 11                      # konstante Verzögerung des Schwellendurchgangs
        assert np.ptp(err) < 5                              # Streuung unter 0,1 ms
    # Klangfarben unterscheiden sich im Onset um weniger als 0,05 ms
    med = [np.median((sg.detect_onsets(sg.render_track(n, 8, ph)) - ph) % sg.BEAT) for n in "ABC"]
    assert np.ptp(med) < 2.4


def test_phase_lock_follows_and_wrap():
    lock = sg.PhaseLock()
    x = sg.render_track("A", 8, sg.BEAT - 3.0)              # Phase dicht am Wrap
    assert lock.update(sg.detect_onsets(x))
    assert abs(sg.wrap(lock.phase - (sg.BEAT - 3.0 + 8.5))) < 2.0
    assert not sg.PhaseLock().update(sg.detect_onsets(np.zeros(FS)))


def _synthetic_run(tmp_path, jump_ms, gap=False):
    """90 s Programm: bis 45 s Kicks von A, danach von B um ``jump_ms`` versetzt; Trace mit Rate 1."""
    n = 90 * FS
    a = sg.render_track("A", 90, 777.0)
    b = sg.render_track("B", 90, 777.0 + jump_ms * 48.0)
    x = np.where(np.arange(n) < 45 * FS, a, b)
    # 100-ms-Überblendung wie im Mixer
    f = np.clip((np.arange(n) - 45 * FS) / (0.1 * FS), 0, 1)
    f = 0.5 - 0.5 * np.cos(np.pi * f)
    x = (1 - f) * a + f * b
    if gap:
        x[60 * FS:60 * FS + 480] = 0.0
    wav = tmp_path / "program-test.wav"
    w = WavWriter(wav, FS, 2, "float32")
    w.write((x, x))
    w.close()
    tr = tmp_path / "program-test.trace.jsonl"
    with open(tr, "w") as fh:
        for i in range(0, n, 2048):
            fh.write(json.dumps({"k": "blk", "i": i, "n": 2048, "Tc": float(i), "r0": 1.0, "r1": 1.0,
                                 "b": 3.0, "wall": i, "drift_ms": 0.1, "xr": 0}) + "\n")
        fh.write(json.dumps({"k": "ev", "i": 0, "wall": 0, "kind": "handoff_effective", "source": "B",
                             "previous": "A", "t_switch": 45 * FS, "Tc_switch": 45 * FS}) + "\n")
    return wav, tr


def test_analysis_measures_phase_jump(tmp_path):
    wav, tr = _synthetic_run(tmp_path, jump_ms=7.0)
    res = analyze.analyze_profile(wav, tr, "clean")
    assert res["handovers_effective"] == 1
    p = res["per_handover"][0]
    assert p["phase_jump_ms"] == pytest.approx(7.0, abs=0.3)
    assert p["gaps"] == 0 and p["max_jump_ratio"] < 1.5
    assert not res["acceptance"]["phase_jump<=1.0ms"] and not res["acceptance"]["passed"]
    assert "Profil" in analyze.markdown([res])


def test_analysis_passes_when_aligned_and_reports_gaps(tmp_path):
    wav, tr = _synthetic_run(tmp_path, jump_ms=0.4)
    res = analyze.analyze_profile(wav, tr, "clean")
    assert res["per_handover"][0]["phase_jump_ms"] == pytest.approx(0.4, abs=0.1)
    assert res["acceptance"]["phase_jump<=1.0ms"]
    (tmp_path / "g").mkdir()
    wav2, tr2 = _synthetic_run(tmp_path / "g", 0.0, gap=True)
    res2 = analyze.analyze_profile(wav2, tr2, "wan120")
    assert res2["summary"]["gaps_total"] == 1 and res2["summary"]["max_gap_ms"] == pytest.approx(10.0, abs=0.1)


def test_phase_measurement_without_handover_is_zero_wherever_the_window_starts(tmp_path):
    """Nullmessung: ohne Übergabe misst ``phase_jump`` 0, auch wenn ein Fenster mitten in einem Kick oder in
    seinem Ausklang beginnt. Vorher galt der erste Schwellendurchgang im Fenster als Onset (ohne Ruhephase
    davor) und verschob das Kreismittel um bis zu 0,8 ms (S3904)."""
    n = 120 * FS
    x = sg.render_track("B", 120, 777.0)
    wav = tmp_path / "program-null.wav"
    w = WavWriter(wav, FS, 2, "float32")
    w.write((x, x))
    w.close()
    tm = analyze.TimeMap([{"i": i, "Tc": float(i)} for i in range(0, n, 2048)])
    mid = x.astype(np.float32)
    jumps = [analyze.phase_jump(mid, tm, c)["jump_ms"] for c in range(30 * FS, 90 * FS, int(0.37 * FS))]
    assert max(abs(j) for j in jumps) < 0.02, max(abs(j) for j in jumps)
    nulls = analyze.null_jumps(mid, tm, [])
    assert len(nulls) == 1 and abs(nulls[0]["jump_ms"]) < 0.02


def test_runner_dry_run_shows_plan_without_docker():
    out = subprocess.run([sys.executable, str(ITEST / "run.py"), "--dry-run"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    txt = out.stdout
    for needle in ("clean", "wan40", "wan120", "delay 40ms 5ms", "loss 0.5%", "docker compose", "10 Übergaben",
                   "POST /handover", "analyze"):
        assert needle in txt, needle
    one = subprocess.run([sys.executable, str(ITEST / "run.py"), "--dry-run", "--profiles", "clean", "--handovers", "2"],
                         capture_output=True, text=True, timeout=60)
    assert one.returncode == 0 and "wan40" not in one.stdout and "2 Übergaben" in one.stdout


FAKE_DOCKER = """#!/bin/sh
# docker-Ersatz: Stack startet, clientA ist sofort beendet (wie wan120 in Run 42515682729)
case "$*" in
  *" ps "*) echo clientA ;;
  *" config --services"*) printf 'server\\nclientA\\n' ;;
  *" logs "*) for i in $(seq 1 300); do echo "$4 Zeile $i"; done; echo "letzte Zeile von $*" ;;
  *) : ;;
esac
exit 0
"""


def test_runner_aborted_profile_dumps_logs_and_writes_analysis(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "docker").write_text(FAKE_DOCKER)
    (bindir / "docker").chmod(0o755)
    out = tmp_path / "out"
    env = {**__import__("os").environ, "PATH": f"{bindir}:" + __import__("os").environ["PATH"]}
    r = subprocess.run([sys.executable, str(ITEST / "run.py"), "--profiles", "clean,wan120", "--handovers", "1",
                        "--out", str(out), "--api-port", "9", "--no-build"],
                       capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 1, r.stderr
    assert "Container beendet: clientA" in r.stdout
    assert "Compose-Log b2bit-clean/clientA (letzte 200 Zeilen)" in r.stdout
    assert "letzte Zeile von" in r.stdout and "--tail 200 clientA" in r.stdout
    for prof in ("clean", "wan120"):                      # der Abbruch des ersten Profils hält das zweite nicht auf
        res = json.loads((out / prof / "analysis.json").read_text())
        assert res["profile"] == prof and "Container beendet" in res["error"]
        assert res["acceptance"]["passed"] is False and res["completed"] is False
        assert (out / prof / "logs" / "clientA.log").exists()
    assert len(json.loads((out / "results.json").read_text())) == 2
    assert "abgebrochen" in (out / "report.md").read_text()


def test_mark_incomplete_keeps_measurements(tmp_path):
    wav, tr = _synthetic_run(tmp_path, jump_ms=0.4)
    res = analyze.mark_incomplete(analyze.analyze_profile(wav, tr, "clean"), "TimeoutError: x")
    assert res["per_handover"][0]["phase_jump_ms"] == pytest.approx(0.4, abs=0.1)
    assert res["acceptance"]["passed"] is False and res["error"] == "TimeoutError: x"
    assert "Abgebrochen" in analyze.markdown([res])
