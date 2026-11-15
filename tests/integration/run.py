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

Profil ``noname`` (S3904): nur Server und ein JackTrip-Client **ohne** ``--remotename`` (der Hub benennt ihn nach
der Adresse, ``__ffff_<ip>``), der erst ``--late-join-s`` (20 s) nach dem Server startet - wie ein echter DJ, der
sich in den laufenden Dienst einwählt. Der Orchestrator findet den Namen über ``/status``, setzt ihn live und
prüft nach ``--noname-hold-s``, dass das Programm seinen Ton führt (Pegel je Haltezeit, ``underruns`` steigt nicht).

Profile ``gone`` und ``gone_hard`` (S4006): wie ``noname``, danach wird der Client beendet - ``gone`` sauber (SIGINT,
JackTrip sendet das Stopp-Paket), ``gone_hard`` hart (SIGKILL des Containers, kein Stopp-Paket, die Ports bleiben
bis zum UDP-Timeout des Hubs). ``/status`` muss ``no_source=true`` (``live`` leer, Ereignis ``source_lost``) und
``recording_info.state == "waiting"`` melden; die Quelle darf nicht mehr als verbunden gelten. Die Fristen folgen den
eingestellten Zeiten: Karenz ``B2B_SOURCE_GRACE_S`` (5 s) bzw. Stille-Frist ``B2B_SOURCE_SILENCE_S`` (12 s) plus
Zuschlag 5 s, die Aufnahme folgt 2 s Nachlauf später. Diese Profile prüfen nur den Status, nicht das Audio.

Bricht ein Profil ab (Zeitüberschreitung, beendeter Container, Fehler der Auswertung), gibt der Lauf die
letzten ``--log-lines`` (200) Zeilen der Compose-Logs je Container aus (auch nach ``<profil>/logs/``),
wertet den Mitschnitt aus, soweit er reicht, und schreibt ``analysis.json`` mit Feld ``error``; die übrigen
Profile laufen weiter, ``results.json`` und ``report.md`` entstehen in jedem Fall.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPOSE = HERE / "docker-compose.yml"

# ``rate 1000mbit``: netem plant jedes Paket frühestens hinter dem vorigen ein. Ohne ``rate`` sortiert netem
# nach individueller Verzögerung - bei 5 ms Jitter und 5,33 ms Paketabstand kämen ~23 % der JackTrip-Pakete
# hinter ihrem Nachfolger an und gingen als Lücke verloren (wan40 in Run 42515682729: ~41 Lücken/s zu
# 3-5 Paketen, Relay ohne Unterlauf). Gewolltes Umordnen steht ausdrücklich da (``reorder`` in wan120).
PROFILES = {
    "clean": "",
    "wan40": "delay 40ms 5ms distribution normal rate 1000mbit",
    "wan120": "delay 120ms 20ms distribution normal loss 0.5% reorder 5% 50% rate 1000mbit",
    "noname": "",
    "gone": "",
    "gone_hard": "",
}
# Profile gone / gone_hard (S4006): der einzige Client geht, der Server muss das melden.
#   gone       sauber: SIGINT -> jacktrip sendet das Stopp-Paket, der Hub entfernt die Ports sofort;
#              Frist = Karenz B2B_SOURCE_GRACE_S + Zuschlag
#   gone_hard  hart: SIGKILL des Containers, kein Stopp-Paket; Ports bleiben bis zum UDP-Timeout des Hubs (10 s),
#              Frist = Stille-Frist B2B_SOURCE_SILENCE_S + Zuschlag
# Die Fristen leiten sich aus den eingestellten Zeiten ab (dieselben Werte gehen als ITEST_* an den Server).
GONE = {"gone": "SIGINT", "gone_hard": "SIGKILL"}
SOLO = {"noname"} | set(GONE)   # Profile mit einem einzigen, spät verbundenen Client ohne Remote Name
GRACE_S = 5.0               # B2B_SOURCE_GRACE_S im Teststack (= Vorgabe)
SILENCE_S = 12.0            # B2B_SOURCE_SILENCE_S im Teststack (= Vorgabe)
SAFETY_S = 5.0              # Zuschlag: Router-Takt 0,5 s, Poll 1 s, Worker-Block, geteilte CI-Runner
RECORD_HOLD_S = 2.0         # Nachlauf der Aufnahme nach dem Verlust der live-Quelle (recorder.HOLD_S)


def gone_deadline(name: str) -> float:
    """Frist ab dem Beenden des Clients bis no_source=true (Aufnahme 'waiting' kommt RECORD_HOLD_S später)."""
    return (SILENCE_S if name == "gone_hard" else GRACE_S) + SAFETY_S
DJS = ["A", "B", "C"]


class ProfileAbort(RuntimeError):
    """Profil vorzeitig beendet (Zeitüberschreitung, Container weg)."""


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
            self.check_alive()
            time.sleep(every)
        raise TimeoutError(f"Zeitüberschreitung: {what} ({timeout:.0f} s)")

    # ---------------------------------------------------------------- Container
    def compose_cmd(self) -> list[str]:
        prof = ["--profile", "noname"] if getattr(self, "solo", False) else []
        return ["docker", "compose", "-f", str(COMPOSE), "-p", self.proj] + prof

    def check_alive(self) -> None:
        """Bricht ab, wenn ein Container des Profils beendet ist (statt bis zur Zeitüberschreitung zu warten)."""
        if self.a.dry_run or not getattr(self, "proj", None):
            return
        r = subprocess.run(self.compose_cmd() + ["ps", "-a", "--status", "exited", "--services"],
                           env={**os.environ, **self.env}, capture_output=True, text=True, check=False)
        dead = [x for x in r.stdout.split() if x and x not in getattr(self, "expected_exit", ())]
        if dead:
            raise ProfileAbort(f"Container beendet: {', '.join(dead)}")

    def dump_logs(self, d: Path, lines: int) -> None:
        """Letzte ``lines`` Zeilen je Container ausgeben und unter ``d/logs/<dienst>.log`` ablegen."""
        env = {**os.environ, **self.env}
        r = subprocess.run(self.compose_cmd() + ["config", "--services"], env=env, capture_output=True,
                           text=True, check=False)
        (d / "logs").mkdir(parents=True, exist_ok=True)
        for svc in [x for x in r.stdout.split() if x]:
            lg = subprocess.run(self.compose_cmd() + ["logs", "--no-color", "--tail", str(lines), svc], env=env,
                                capture_output=True, text=True, check=False)
            text = lg.stdout + lg.stderr
            (d / "logs" / f"{svc}.log").write_text(text)
            print(f"::group::Compose-Log {self.proj}/{svc} (letzte {lines} Zeilen)", flush=True)
            print(text.rstrip(), flush=True)
            print("::endgroup::", flush=True)

    # ---------------------------------------------------------------- ein Profil
    def profile(self, name: str, netem: str) -> dict:
        a = self.a
        d = self.out / name
        self.proj = f"b2bit-{name}"
        self.solo = name in SOLO
        self.profile_name = name
        self.client_name: str | None = None
        self.gone: dict | None = None
        self.expected_exit: set[str] = set()
        self.env = env = {"NETEM": netem, "ITEST_OUT": str(d), "ITEST_API_PORT": str(a.api_port),
                          "ITEST_JACKTRIP_QUEUE": str(a.jacktrip_queue),
                          "ITEST_SOURCE_GRACE_S": f"{GRACE_S:g}", "ITEST_SOURCE_SILENCE_S": f"{SILENCE_S:g}"}
        if name in GONE:
            env["ITEST_RECORD_ONLY_LIVE"] = "true"      # sonst kann die Aufnahme nie 'waiting' melden
        compose = self.compose_cmd()
        error: str | None = None
        self.say(f"\n=== Profil {name}: netem {netem or '(keine Störung)'}, JackTrip-Queue des Hubs {a.jacktrip_queue} ===")
        if not a.dry_run:
            for sub in ("rec", "ctl"):
                (d / sub).mkdir(parents=True, exist_ok=True)
                os.chmod(d / sub, 0o777)
            (d / "ctl" / "next").write_text("")
            os.chmod(d / "ctl" / "next", 0o666)
        status_log = d / "status.jsonl"
        stop = threading.Event()
        reqs: list[dict] = []
        try:
            if self.solo:
                self._solo_steps(compose, env, status_log, stop)
            else:
                self.say("  [1] Stack starten")
                self.sh(compose + ["up", "-d"] + ([] if a.no_build else ["--build"]), env)
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
                self.check_alive()
        except Exception as e:          # Profil abgebrochen: Logs sichern (Container laufen noch), Rest auswerten
            error = f"{type(e).__name__}: {e}"
            self.say(f"  !! Profil {name} abgebrochen: {error}")
            traceback.print_exc()
            if not a.dry_run:
                self.dump_logs(d, a.log_lines)
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
        if name in GONE:
            res = self._gone_result(name, error)
            res["jacktrip_queue"] = str(a.jacktrip_queue)
            (d / "analysis.json").write_text(json.dumps(res, indent=1))
            return res
        res = None
        wavs = sorted((d / "rec").glob("program-*.wav"))
        if not wavs:
            error = (error + "; " if error else "") + f"kein Mitschnitt in {d / 'rec'}"
        else:
            try:
                res = analyze.analyze_profile(wavs[-1], wavs[-1].with_suffix(".trace.jsonl"), name, status_log, reqs)
            except Exception as e:
                traceback.print_exc()
                error = (error + "; " if error else "") + f"Auswertung: {type(e).__name__}: {e}"
        if res is None:
            res = analyze.incomplete(name, error or "unbekannt", reqs)
        elif error:
            res = analyze.mark_incomplete(res, error)
        if self.solo:
            res["client_name"] = self.client_name
            if not error and not a.dry_run:
                ok = self.client_name is not None and self.client_name not in DJS
                res["acceptance"] = dict(res["acceptance"], client_without_remotename=ok)
                res["acceptance"]["passed"] = all(v for k, v in res["acceptance"].items() if k != "passed")
        res["jacktrip_queue"] = str(a.jacktrip_queue)
        (d / "analysis.json").write_text(json.dumps(res, indent=1))
        return res

    def _solo_steps(self, compose: list[str], env: dict, status_log: Path, stop: threading.Event) -> None:
        a = self.a
        build = [] if a.no_build else ["--build"]
        self.say("  [1] Server starten (ohne Clients)")
        self.sh(compose + ["up", "-d"] + build + ["server"], env)
        if not a.dry_run:
            self.wait("Server-API", lambda: self.http("GET", "/healthz")[0] == 200, 120)
            threading.Thread(target=self._log_status, args=(status_log, stop), daemon=True).start()
        self.say(f"  [2] {a.late_join_s:.0f} s warten, dann Client ohne --remotename starten (später Join)")
        if not a.dry_run:
            time.sleep(a.late_join_s)
        self.sh(compose + ["up", "-d"] + build + ["clientN"], env)
        self.say("  [3] Namen des Clients aus /status (nicht A/B/C), POST /handover an ihn")
        if a.dry_run:
            self.say(f"  [4] Haltezeit {(a.gone_hold_s if self.profile_name in GONE else a.noname_hold_s):.0f} s")
            if self.profile_name in GONE:
                self._say_gone()
            return
        names = self.wait("Client ohne Remote Name verbunden", lambda: self._present() - set(DJS), a.connect_timeout_s)
        self.client_name = sorted(names)[0]
        self.say(f"      Client erscheint als {self.client_name!r}")
        self.wait(f"{self.client_name} live",
                  lambda: self.http("POST", "/handover", {"to": self.client_name})[0] in (200, 409)
                  and self.http("GET", "/status")[1].get("live") == self.client_name, 30)
        hold = a.gone_hold_s if self.profile_name in GONE else a.noname_hold_s
        self.say(f"  [4] Haltezeit {hold:.0f} s")
        time.sleep(hold)
        self.check_alive()
        if self.profile_name in GONE:
            self._gone_steps(compose, env)

    def _say_gone(self) -> None:
        n = self.profile_name
        d = gone_deadline(n)
        self.say(f"  [5] Client beenden ({GONE[n]}), innerhalb {d:.0f} s no_source=true, innerhalb {d + RECORD_HOLD_S + 1:.0f} s "
                 f"Aufnahme 'waiting' erwartet")

    def _gone_steps(self, compose: list[str], env: dict) -> None:
        """Client beenden (gone: SIGINT, gone_hard: SIGKILL), dann: no_source, live leer, Ereignis, Aufnahme 'waiting'."""
        n = self.profile_name
        d = gone_deadline(n)
        self._say_gone()
        self.expected_exit = {"clientN"}        # beendeter Client ist hier gewollt (check_alive)
        t0 = time.monotonic()                   # vor dem Signal: die Frist schließt den Aufruf ein
        self.sh(compose + ["kill", "-s", GONE[n], "clientN"], env, check=False)
        g: dict = {"mode": GONE[n], "deadline_s": d, "waiting_deadline_s": d + RECORD_HOLD_S + 1,
                   "no_source_s": None, "waiting_s": None}
        while time.monotonic() - t0 < g["waiting_deadline_s"] + SAFETY_S:
            code, js = self.http("GET", "/status", timeout=3)
            if code == 200:
                el = round(time.monotonic() - t0, 1)
                if js.get("no_source") and js.get("live") is None and g["no_source_s"] is None:
                    g["no_source_s"] = el
                    g["source_gone"] = self.client_name not in (js.get("sources") or {})
                    lost = [e for e in js.get("events", []) if e.get("kind") == "source_lost"
                            and e.get("source") == self.client_name and e.get("was_live")]
                    g["source_lost_event"] = bool(lost)
                    g["source_lost_reason"] = lost[-1].get("reason") if lost else None
                if (js.get("recording_info") or {}).get("state") == "waiting" and g["waiting_s"] is None:
                    g["waiting_s"] = el
                if g["no_source_s"] is not None and g["waiting_s"] is not None:
                    break
            time.sleep(1.0)
        self.gone = g

    def _gone_result(self, name: str, error: str | None) -> dict:
        g = self.gone or {}
        lim = g.get("deadline_s", gone_deadline(name))
        lim_w = g.get("waiting_deadline_s", lim + RECORD_HOLD_S + 1)
        acc = {"no_source_in_time": g.get("no_source_s") is not None and g["no_source_s"] <= lim,
               "recording_waiting_in_time": g.get("waiting_s") is not None and g["waiting_s"] <= lim_w,
               "source_not_listed": bool(g.get("source_gone")),
               "source_lost_event": bool(g.get("source_lost_event")),
               "client_without_remotename": self.client_name is not None and self.client_name not in DJS}
        res = {"profile": name, "completed": error is None, "error": error, "handovers_effective": 0,
               "handovers_refused": 0, "fallbacks": 0, "per_handover": [], "summary": {}, "series": [],
               "client_name": self.client_name, "gone": g}
        if error:
            acc = {k: False for k in acc}
        acc["completed"] = error is None
        acc["passed"] = all(acc.values())
        res["acceptance"] = acc
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
            self.check_alive()
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
    ap.add_argument("--late-join-s", type=float, default=20.0,
                    help="Profil noname: so lange nach dem Server startet der Client (später Join, S3904)")
    ap.add_argument("--noname-hold-s", type=float, default=90.0, help="Profil noname: Haltezeit nach dem Live-Setzen")
    ap.add_argument("--gone-hold-s", type=float, default=30.0, help="Profile gone/gone_hard: Haltezeit vor dem Beenden")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--api-port", type=int, default=18090)
    ap.add_argument("--token", default=os.environ.get("B2B_API_TOKEN", ""))
    ap.add_argument("--jacktrip-queue", default="12",
                    help="Jitterpuffer des Hubs (JACKTRIP_QUEUE): Slots oder auto. Vorgabe 12: mit auto verschiebt "
                         "JackTrip den Feed bei Jitter um ganze Slots (5,3 ms), siehe README")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--no-fail", action="store_true", help="Exit 0 auch bei verfehlter Abnahme")
    ap.add_argument("--log-lines", type=int, default=200, help="Compose-Log-Zeilen je Container bei Abbruch")
    a = ap.parse_args(argv)
    names = [p.strip() for p in a.profiles.split(",") if p.strip()]
    for n in names:
        if n not in PROFILES:
            ap.error(f"unbekanntes Profil {n!r}")
    r = Runner(a)
    total = sum(((a.late_join_s + a.gone_hold_s + gone_deadline(n) + RECORD_HOLD_S + 2 * SAFETY_S) if n in GONE else (a.late_join_s + a.noname_hold_s) if n in SOLO else a.warmup_s + a.handovers * a.hold_s) + a.settle_s
                for n in names) / 60
    print(f"Integrationstest: Profile {names}, je {a.handovers} Übergaben in Folge (A→B→C→…), "
          f"{a.hold_s:.0f} s je Übergabe; geschätzt {total:.0f} min ohne Image-Bau")
    print(f"Ausgabe: {r.out}   (results.json, report.md, je Profil rec/*.wav + *.trace.jsonl, status.jsonl)")
    results = []
    for n in names:
        try:
            results.append(r.profile(n, PROFILES[n]))
        except Exception as e:      # Fehler des Harness selbst: Profil als abgebrochen führen, weiter
            traceback.print_exc()
            if a.dry_run:
                raise
            import analyze
            res = analyze.incomplete(n, f"{type(e).__name__}: {e}")
            (r.out / n).mkdir(parents=True, exist_ok=True)
            (r.out / n / "analysis.json").write_text(json.dumps(res, indent=1))
            results.append(res)
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
