#!/usr/bin/env python3
"""Integrationstest-Orchestrator (Scheibe 2a): Server + 3 Test-DJs unter simulierter Netzstörung.

    python tests/integration/run.py --dry-run                 # Ablauf zeigen, nichts starten
    python tests/integration/run.py                           # alle Profile, je 10 Übergaben
    python tests/integration/run.py --profiles clean --handovers 3 --hold-s 45

Je Profil: Compose-Stack hochfahren (Netzstörung per ``NETEM`` im Client-Container), warten bis alle drei
DJs verbunden sind, A als ersten Live-DJ setzen, dann ``N`` Übergaben in Folge (A→B→C→A…): kurz vor der
Übergabe schreibt der Orchestrator den Übernehmenden nach ``ctl/next`` (er mischt dann seine Kicks phasenrichtig
zum gehörten Feed dazu), wartet auf die RTT-Einrastung, ruft ``POST /handover`` (bei ``buffer_too_low`` mit
der genannten Wartezeit erneut) und lässt den Rest der Haltezeit laufen. Danach Auswertung des
Programm-Mitschnitts (``analyze.py``): Ergebnis als ``results.json`` und ``report.md``.
Exit 0, wenn alle Abnahmekriterien aller Profile erfüllt sind (``--no-fail``: immer 0).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPOSE = HERE / "docker-compose.yml"

PROFILES = {
    "clean": "",
    "wan40": "delay 40ms 5ms distribution normal",
    "wan120": "delay 120ms 20ms distribution normal loss 0.5% reorder 5% 50%",
}
DJS = ["A", "B", "C"]


class Runner:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.out = Path(a.out).resolve()
        self.api = f"http://127.0.0.1:{a.api_port}"

    # ---------------------------------------------------------------- Hilfen
    def sh(self, cmd: list[str], env: dict | None = None, check: bool = True, timeout: int | None = None):
        line = " ".join(cmd)
        if self.a.dry_run:
            print(f"    $ {line}")
            return None
        return subprocess.run(cmd, env={**os.environ, **(env or {})}, check=check, timeout=timeout,
                              capture_output=False)

    def say(self, msg: str) -> None:
        print(msg, flush=True)

    def http(self, method: str, path: str, body: dict | None = None, timeout: float = 10.0):
        req = urllib.request.Request(self.api + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        if self.a.token:
            req.add_header("Authorization", f"Bearer {self.a.token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, {"raw": raw.decode(errors="replace")}
        except (urllib.error.URLError, OSError, ConnectionError) as e:
            return 0, {"error": str(e)}

    def wait(self, what: str, fn, timeout: float, every: float = 2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = fn()
            if v:
                return v
            time.sleep(every)
        raise TimeoutError(f"Zeitüberschreitung: {what} ({timeout:.0f} s)")

    # ---------------------------------------------------------------- ein Profil
    def profile(self, name: str, netem: str) -> dict:
        a = self.a
        d = self.out / name
        proj = f"b2bit-{name}"
        env = {"NETEM": netem, "ITEST_OUT": str(d), "ITEST_API_PORT": str(a.api_port)}
        compose = ["docker", "compose", "-f", str(COMPOSE), "-p", proj]
        self.say(f"\n=== Profil {name}: netem {netem or '(keine Störung)'} ===")
        if not a.dry_run:
            for sub in ("rec", "ctl"):
                (d / sub).mkdir(parents=True, exist_ok=True)
                os.chmod(d / sub, 0o777)
            (d / "ctl" / "next").write_text("")
            os.chmod(d / "ctl" / "next", 0o666)
        self.say("  [1] Stack starten")
        self.sh(compose + ["up", "-d"] + ([] if a.no_build else ["--build"]), env)
        status_log = d / "status.jsonl"
        stop = threading.Event()
        reqs: list[dict] = []
        try:
            self.say("  [2] warten auf /healthz und auf die DJs A, B, C (Clients present)")
            if not a.dry_run:
                self.wait("Server-API", lambda: self.http("GET", "/healthz")[0] == 200, 120)
                self.wait("DJs verbunden", lambda: self._present() >= set(DJS), a.connect_timeout_s)
            self.say(f"  [3] POST /handover {{to: A}} (erster Live-DJ), Einlaufzeit {a.warmup_s:.0f} s")
            if not a.dry_run:
                self.wait("A live", lambda: self.http("POST", "/handover", {"to": "A"})[0] in (200, 409), 30)
                threading.Thread(target=self._log_status, args=(status_log, stop), daemon=True).start()
                time.sleep(a.warmup_s)
            for k in range(1, a.handovers + 1):
                frm, to = DJS[(k - 1) % 3], DJS[k % 3]
                self.say(f"  [4.{k}] Übergabe {k}/{a.handovers}: {frm} → {to}  (ctl/next={to}, {a.lead_s:.0f} s Vorlauf, "
                         f"POST /handover, dann bis {a.hold_s:.0f} s Haltezeit)")
                if a.dry_run:
                    continue
                reqs.append(self._handover(d, frm, to))
            self.say(f"  [5] Nachlauf {a.settle_s:.0f} s, Mitschnitt schließen")
            if not a.dry_run:
                time.sleep(a.settle_s)
        finally:
            stop.set()
            if not a.dry_run:
                subprocess.run(compose + ["logs", "--no-color", "--tail", "400"], env={**os.environ, **env},
                               stdout=open(d / "compose.log", "w"), stderr=subprocess.STDOUT, check=False)
            self.say("  [6] Stack beenden (SIGTERM an den Server schließt die WAV-Datei)")
            self.sh(compose + ["stop", "-t", "15"], env, check=False)
            self.sh(compose + ["down", "-v", "--remove-orphans"], env, check=False)
        self.say("  [7] analyze: WAV + Trace → Phasensprung, Lücken, Klicks, xruns, Puffer/Rate")
        if a.dry_run:
            return {"profile": name, "dry_run": True}
        import analyze
        wavs = sorted((d / "rec").glob("program-*.wav"))
        if not wavs:
            raise RuntimeError(f"kein Mitschnitt in {d / 'rec'} (siehe {d / 'compose.log'})")
        res = analyze.analyze_profile(wavs[-1], wavs[-1].with_suffix(".trace.jsonl"), name, status_log, reqs)
        (d / "analysis.json").write_text(json.dumps(res, indent=1))
        return res

    def _present(self) -> set[str]:
        code, js = self.http("GET", "/status")
        return {c["name"] for c in js.get("clients", []) if c.get("present")} if code == 200 else set()

    def _log_status(self, path: Path, stop: threading.Event) -> None:
        with open(path, "w") as fh:
            while not stop.wait(2.0):
                code, js = self.http("GET", "/status", timeout=3)
                if code == 200:
                    js["t_mono"] = time.monotonic()
                    fh.write(json.dumps(js) + "\n")
                    fh.flush()

    def _handover(self, d: Path, frm: str, to: str) -> dict:
        a = self.a
        t0 = time.monotonic()
        (d / "ctl" / "next").write_text(to)
        time.sleep(a.lead_s)
        rec = {"from": frm, "to": to, "t_next_mono": t0, "attempts": 0, "accepted": False}
        try:
            self.wait(f"RTT von {to} eingerastet",
                      lambda: self.http("GET", "/status")[1].get("sources", {}).get(to, {}).get("locked"), 90)
        except TimeoutError as e:
            rec["lock_timeout"] = str(e)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            rec["attempts"] += 1
            code, js = self.http("POST", "/handover", {"to": to})
            if code == 200:
                rec.update(accepted=True, t_request_mono=time.monotonic(), result=js)
                break
            detail = js.get("detail", js) if isinstance(js, dict) else {}
            reason = detail.get("reason") if isinstance(detail, dict) else None
            rec["last_refusal"] = detail
            wait_s = detail.get("wait_s") if isinstance(detail, dict) else None
            time.sleep(min(30.0, (wait_s or 5.0) + 1.0))
        if rec["accepted"]:
            try:
                self.wait(f"{to} live", lambda: self.http("GET", "/status")[1].get("live") == to, 60, 1.0)
                rec["t_live_mono"] = time.monotonic()
            except TimeoutError as e:
                rec["live_timeout"] = str(e)
        rest = a.hold_s - (time.monotonic() - t0)
        if rest > 0:
            time.sleep(rest)
        return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="Ablauf anzeigen, nichts ausführen")
    ap.add_argument("--profiles", default=",".join(PROFILES), help="kommagetrennt: " + ", ".join(PROFILES))
    ap.add_argument("--handovers", type=int, default=10, help="Übergaben je Profil (Vorgabe 10)")
    ap.add_argument("--hold-s", type=float, default=60.0, help="Zeit je Übergabe-Zyklus")
    ap.add_argument("--lead-s", type=float, default=30.0, help="Vorlauf: Übernehmender mischt, bevor übergeben wird")
    ap.add_argument("--warmup-s", type=float, default=45.0)
    ap.add_argument("--settle-s", type=float, default=20.0)
    ap.add_argument("--connect-timeout-s", type=float, default=240.0)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--api-port", type=int, default=18090)
    ap.add_argument("--token", default=os.environ.get("B2B_API_TOKEN", ""))
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--no-fail", action="store_true", help="Exit 0 auch bei verfehlter Abnahme")
    a = ap.parse_args(argv)
    names = [p.strip() for p in a.profiles.split(",") if p.strip()]
    for n in names:
        if n not in PROFILES:
            ap.error(f"unbekanntes Profil {n!r}")
    r = Runner(a)
    total = (a.warmup_s + a.handovers * a.hold_s + a.settle_s) * len(names) / 60
    print(f"Integrationstest: Profile {names}, je {a.handovers} Übergaben in Folge (A→B→C→…), "
          f"{a.hold_s:.0f} s je Übergabe; geschätzt {total:.0f} min ohne Image-Bau")
    print(f"Ausgabe: {r.out}   (results.json, report.md, je Profil rec/*.wav + *.trace.jsonl, status.jsonl)")
    results = []
    for n in names:
        results.append(r.profile(n, PROFILES[n]))
    if a.dry_run:
        print("\n[Trockenlauf] Danach: results.json + report.md schreiben; Exit 1, wenn eine Abnahme verfehlt wird.")
        return 0
    import analyze
    (r.out / "results.json").write_text(json.dumps(results, indent=1))
    (r.out / "report.md").write_text(analyze.markdown(results))
    print((r.out / "report.md").read_text())
    ok = all(x["acceptance"]["passed"] for x in results)
    return 0 if (ok or a.no_fail) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.exit(main())
