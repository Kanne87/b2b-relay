# b2b-relay

Online-Back-to-Back für DJs über das Internet – unabhängig von der DJ-Software.

## Prinzip (Staffelstab)

Nie sind zwei DJs gleichzeitig in Echtzeit live. Das Programm entsteht immer bei dem DJ, der gerade mixt:

1. Der Server führt eine Programm-Zeitachse T und sendet mit festem Puffer D (1–3 s) an die Plattform.
2. Der übernehmende DJ bekommt das Programm P(T) latenzarm als Rückkanal, legt es auf einen freien Kanal und mixt seinen Track lokal phasengenau dagegen.
3. Sein Master (Programm + neuer Track) kommt um seine Round-Trip-Zeit RTT versetzt zurück.
4. Der Relay-Mixer misst RTT je DJ, fädelt den Feed um −RTT versetzt auf T ein und schaltet bei der Übergabe mit kurzem Crossfade um. Bedingung: D > RTT + Jitter-Reserve.
5. Rollentausch.

## Softwareunabhängig

Der Server sieht nur Audio. BPM, Downbeat und Taktposition kommen aus Beat-Erkennung auf dem Programm-Audio, nicht aus Metadaten der DJ-Software. Weil der Server die Laufzeit je DJ kennt, zeigt die Weboberfläche jedem DJ einen latenzkorrigierten Takt- und Phrasenzähler.

## Scheiben

1. Kern: Relay-Mixer-Algorithmus (Zeitachse, RTT-Messung, Ausrichtung, Umschaltung) mit deterministischer Offline-Simulation. **← dieser Stand**
2. Live-Audio: Transport (JackTrip-Hub/JACK) auf dem Server, Test mit künstlicher Netzverzögerung (tc netem).
3. Beat-Analyse + WebSocket + Weboberfläche (BPM, Takt/Phrase latenzkorrigiert, Live-Anzeige, Übergabe-Knopf, Leitungsqualität).
4. Video/OBS, Overlay, Ausspielung Twitch (DJ Program).

---

# Scheibe 1: Kern und Simulation

Python ≥ 3.11, Laufzeit-Abhängigkeit nur `numpy`. `scipy` wird nur von der Simulation gebraucht.

```bash
pip install -e .[dev]     # numpy, scipy, pytest
pytest -q                 # ca. 1 min (davon ~25 s der 10-Minuten-Drifttest)
```

## Architektur

```
src/b2b_relay/
  timeline.py   SampleRing (Ring mit absolutem Index), frac_read (kubische Bruchteil-Interpolation)
  rtt.py        gcc_phat (Versatz per GCC-PHAT, Konfidenz), LagTracker (Nachführung, Halten)
  mixer.py      RelayMixer, RelayConfig, HandoffResult, MixerStatus
  sim/          synth (DnB-Tracks), plan (Drehbuch), dj (DJ-Modell), transport (Paketmodell),
                analysis (Kick-Phase, Lücken, Sprünge), scenario (Lauf + Report), __main__ (CLI)
```

**Zwei Zeitachsen.** `w` = Wanduhr (Eingangsindex je Quelle, zählt mit jedem `process`), `T` = Programmzeit.
Quelle X liefert den Programminhalt T zur Wanduhr `w = T + lag_X`. Die erste aktive Quelle definiert
die Achse (lag 0). Das Programm wird nie als Block gespeichert, sondern beim Lesen aus den Quell-Ringen
zusammengesetzt: `P(T) = Σ gewicht_k(T) · feed_k(T + lag_k)` (kubische Interpolation, Driftrate
eingerechnet). Ausgegeben wird `P(T)` für `T = w − D`, also mit festem Puffer D (Standard 2,0 s).
Weil das Lesen D nach dem Eintreffen geschieht, wirken Umschaltpunkte, bessere Lag-Schätzungen und
der Fallback auch rückwirkend innerhalb des Puffers.

**RTT-Schätzung** (`rtt.py`, pro wartender Quelle alle 0,5 s): GCC-PHAT des Feed-Fensters (4 s) gegen das
Programm, Suchbereich bis 1,5·D. Konfidenz aus Peak-to-Sidelobe-Ratio und Abstand zum zweiten Peak.
Der `LagTracker` nimmt nur Messungen ab Konfidenz 0,5 an, rastet nach zwei stimmigen Messungen ein, passt
Lag und Driftrate per Geradenanpassung über die letzten 20 s an (±500 ppm begrenzt), weist
Ausreißer ab (Neuaufsetzen erst nach drei stimmigen, sehr sicheren Ausreißern) und **hält** bei fehlender
Messung die letzte Schätzung samt Rate. Die Rate wird in die nächste Messung zurückgerechnet (das Fenster wird
entzerrt), damit der Korrelationspeak bei ±100 ppm nicht über ~10 Samples verschmiert. Die aktive Quelle wird nicht
gemessen (ihr Feed *ist* das Programm, die Messung wäre zirkulär); ihre Schätzung läuft mit der zuletzt
bestimmten Rate weiter.

**Übergabe** (`request_handoff(ziel, at=None)`): sofort (am Live-Rand des Programms) oder zur Programmzeit `at`
(Sekunden). Crossfade 50–200 ms (`fade_s`), raised-cosine mit Gewichtssumme 1. Verweigert mit Code und Klartext:
`rtt_exceeds_buffer` (Lag des Ziels + Reserve > D), `not_locked`, `estimate_stale`, `target_silent`, `too_late`
(Zeitpunkt liegt schon vor dem Ausgaberand), `out_of_order`, `already_active`, `unknown_source`. Ist die Schätzung
noch nicht eingerastet, prüft der Mixer gegen den RTT-Hinweis aus `register_source(name, rtt_hint_s=…)`.

**Fallback.** Liefert die aktive Quelle länger als `silence_timeout_s` (0,25 s) Stille (< −60 dBFS), schaltet der
Mixer auf die vorige Quelle; der Crossfade endet dort, wo die Quelle verstummt ist (rückwirkend im Puffer, kein
Loch). Gibt es keine verwendbare vorige Quelle, bleibt es bei Stille und `status.silent` ist gesetzt; ohne aktive
Quelle gibt `process` Stille aus und `status.no_source` ist gesetzt.

**Rückkanal.** Der übernehmende DJ hört den **Feed der abgebenden Quelle** (`read_program(t0, n, source=…)`),
nicht das Programm: Nach der Übergabe käme das Programm aus seinem eigenen Feed (Rückkopplung). Die abgebende
Quelle muss deshalb bis zum Ende des Übergangs weiterspielen – das ist zugleich der Fallback.

### API (blockweise, ohne globalen Zustand)

```python
from b2b_relay import RelayMixer, RelayConfig
mixer = RelayMixer(RelayConfig(buffer_s=2.0, reserve_s=0.3, fade_s=0.1))
mixer.register_source("A", rtt_hint_s=0.04)
mixer.register_source("B", rtt_hint_s=0.25)
mixer.set_active("A")                          # A definiert die Zeitachse
program = mixer.process({"A": blk_a, "B": blk_b})   # gleich lange float32-Blöcke (256/512 …) → P(T=w−D)
res = mixer.request_handoff("B")               # oder at=<Sekunden Programmzeit>
if not res.accepted: print(res.reason, res.message)
monitor = mixer.read_program(t0, n, source="A")     # Rückkanal für B
mixer.status, mixer.source_info("B"), mixer.events
```

## Simulation

```bash
python -m b2b_relay.sim --handoffs 10 --rtt-a-ms 40 --rtt-b-ms 250 --jitter-ms 15 \
       --loss 0.001 --drift-ppm 50 --seed 1 --out report.json [--wav program.wav]
```

Weitere Schalter: `--human-error-ms` (±5), `--mix-bars` (8), `--buffer-s` (2,0), `--reserve-ms` (300),
`--fade-ms` (100), `--block` (512). Exit-Code 1, wenn ein Abnahmekriterium reißt.

Modell: synthetische 174-BPM-Tracks (Kick/Snare/Hi-Hat/Bass/Stabs, je Track per Seed, Bar-Muster variieren).
Szenario A→B→A…: jeder DJ mischt nach der Übernahme zwei eigene Tracks hinein, der andere hört zu (Rückkanal mit
Downlink-Latenz), startet seinen Track phasengenau mit menschlichem Fehler (Standard ±5 ms gleichverteilt) und
mischt über N Takte (Bass-Swap in der Mitte, Rückkanal per EQ/Fader ausgeblendet). Der Rückkanal ist per Tiefpass
in Bass/Höhen geteilt. Transport: feste Latenz je Richtung = RTT/2 einschließlich Jitter-Puffer, Pakete zu 128 Samples
mit Zufallsverlust und exponentialverteilter Verspätung (verspätet = verloren), Verlust = Nullen. Taktdrift: A läuft
+ppm, B −ppm schnell (Lag wandert mit der Zeit). Die erste Quelle definiert T; der Lag jedes späteren DJs ist
Lag der Programmquelle + eigene RTT + Drift (siehe Grenzen: Lag summiert sich).

**Report** (`report.json`): `summary`, `acceptance`, `link_stats`, `mixer_events` und je Übergabe:

| Feld | Bedeutung |
|---|---|
| `rtt_true_ms`, `rtt_est_ms`, `rtt_error_ms` | wahre/geschätzte RTT (Lag relativ zur Programmquelle), Fehler = Schätzung − Wahrheit |
| `lag_true_ms`, `lag_est_ms` | kumulierter Lag relativ zur Zeitachse |
| `beat_phase_error_ms`, `beat_phase_error_max_abs_ms` | Phasenfehler des Programms an der Übergabe: Kick-Onsets nach dem Crossfade gegen das erwartete Raster, abzüglich der Lage vor der Übergabe (`beat_offset_before_ms`); isoliert den Relay-Anteil |
| `dj_human_error_ms` | modellierter menschlicher Einstartfehler (getrennt) |
| `new_track_phase_error_ms`, `new_track_relay_error_ms` | Phase der Kicks des neuen Tracks nach dem Bass-Swap (= menschlicher + Relay-Fehler) und Rest nach Abzug des menschlichen Fehlers |
| `max_sample_jump`, `max_sample_jump_baseline` | größter Sample-Sprung um den Crossfade (ohne Paketverlust-Stellen) und im Vergleichsfenster davor |
| `gaps`, `gaps_unexplained`, `max_gap_unexplained_ms` | Läufe digitaler Stille > 5 ms im Fenster [−1 s, +3 s] und ob ein modellierter Paketverlust sie erklärt |
| `beat_kicks` | die Einzelmessungen (`lost`: Fenster berührt einen Paketverlust und wurde nicht gewertet) |

Kick-Onset-Messung: Tiefpass 80 Hz (6. Ordnung, nullphasig), Energiehüllkurve; in einem Fenster (−30/+100 ms)
um den erwarteten Kick wird die Hüllkurve des Programms gegen die der Referenz (idealer Track am erwarteten Ort)
korreliert, Maximum mit Parabel-Verfeinerung. Eine reine Schwellen- oder Steilheitssuche war bei dicht liegenden
Kicks und Bass-Welligkeit nicht stabil (Fehlgriffe von 10–30 ms); der Abgleich löst 0,02 ms auf (Test).

## Grenzen (Scheibe 1)

* **Der Lag summiert sich über Übergaben.** Der übernehmende DJ hört das Programm erst, wenn es beim Server
  angekommen ist, und schickt es um seine RTT versetzt zurück: `lag_neu = lag_alt + RTT_neu`. Die Prüfung
  „D > RTT + Reserve“ gilt deshalb für den *kumulierten* Lag. Bei 40/250 ms RTT wächst er in zehn Übergaben auf
  rund 1,45 s; bei D = 2 s und 300 ms Reserve ist dann Schluss. Eine Zeitachse ohne Sprung kann das nicht
  verhindern. Vorschläge in der PR-Beschreibung.
* Mono, 48 kHz. Die Schätzung läuft synchron in `process` (ca. 40–60 ms je Lauf alle 0,5 s) – im JACK-Callback
  muss sie in einen Hintergrund-Thread.
* Schätzung funktioniert gegen einen gleich lauten fremden Track bis ca. −26 dB Programmanteil (4-s-Fenster);
  darunter, oder wenn der Anteil ganz verschwindet, wird die letzte Schätzung samt Driftrate gehalten. Für die
  aktive Quelle gibt es keine Messung; ein Fehler der Driftrate wirkt über die Haltezeit (1 ppm Fehler = 60 µs/min).
* Brauchbare Eindeutigkeit setzt Musik mit Taktvariation voraus; ein exakt periodischer Loop ist mehrdeutig
  (Peaks im Beat-Abstand von 345 ms). Echte Tracks sind in der Simulation nicht getestet.
* Transport nur als Verlustmaske mit konstanter Latenz; keine Paketumordnung, keine Verlust-Verschleierung
  (Verlust = Nullen), Drift konstant in ppm. Kein Codec, kein Stereo.
* Der Fallback setzt voraus, dass die vorige Quelle weiterspielt; ein Rückschalten auf die ausgefallene Quelle
  geschieht nicht automatisch (nur per `request_handoff`).
* Der menschliche Einstartfehler ist relativ zum *geplanten* Raster, nicht zum (um µs verschobenen) gehörten Programm.
* Nicht Teil: Netzwerk/JACK/JackTrip, Beat-Tracking für die Anzeige, WebSocket, Oberfläche, Video, Deploy.
