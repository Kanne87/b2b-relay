# b2b-relay

Online-Back-to-Back für DJs über das Internet – unabhängig von der DJ-Software.

## Prinzip (Staffelstab)

Nie sind zwei DJs gleichzeitig in Echtzeit live. Das Programm entsteht immer bei dem DJ, der gerade mixt:

1. Der Server führt eine Programm-Zeitachse T und sendet mit einem Ausgabepuffer B (Ziel 3 s, elastisch geregelt, siehe Scheibe 1b) an die Plattform.
2. Der übernehmende DJ bekommt das Programm P(T) latenzarm als Rückkanal, legt es auf einen freien Kanal und mixt seinen Track lokal phasengenau dagegen.
3. Sein Master (Programm + neuer Track) kommt um seine Round-Trip-Zeit RTT versetzt zurück.
4. Der Relay-Mixer misst RTT je DJ, fädelt den Feed um −RTT versetzt auf T ein und schaltet bei der Übergabe mit kurzem Crossfade um. Bedingung: Puffer nach der Übergabe ≥ B_min (ohne Regelung: D > RTT + Jitter-Reserve).
5. Rollentausch.

## Softwareunabhängig

Der Server sieht nur Audio. BPM, Downbeat und Taktposition kommen aus Beat-Erkennung auf dem Programm-Audio, nicht aus Metadaten der DJ-Software. Weil der Server die Laufzeit je DJ kennt, zeigt die Weboberfläche jedem DJ einen latenzkorrigierten Takt- und Phrasenzähler.

## Scheiben

1. Kern: Relay-Mixer-Algorithmus (Zeitachse, RTT-Messung, Ausrichtung, Umschaltung) mit deterministischer Offline-Simulation.
   1b. **Elastischer Ausgang** (Puffer-Regelung per Varispeed, siehe unten).
2. Live-Audio: Transport (JackTrip-Hub/JACK) auf dem Server, Test mit künstlicher Netzverzögerung (tc netem).
   2a. **Live-Server** (JACK-Client hinter JackTrip-Hub, Docker, CI, Integrationstest). **← dieser Stand** (Betrieb: Abschnitt „Betrieb“)
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
  control.py    BufferController (Puffer-Regler mit Totband, Rampenbegrenzung, Vorwärtsrechnung)
  varispeed.py  VarispeedReader (Ausgangsstufe mit variabler Rate, Sinc-Interpolation)
  rtt.py        gcc_phat (Versatz per GCC-PHAT, Konfidenz), LagTracker (Nachführung, Halten)
  mixer.py      RelayMixer, RelayConfig, HandoffResult, MixerStatus
  sim/          synth (DnB-Tracks), plan (Drehbuch), dj (DJ-Modell), transport (Paketmodell),
                analysis (Kick-Phase, Lücken, Sprünge), scenario (Lauf + Report), __main__ (CLI)
```

(Die Beschreibung des festen Puffers D in diesem Abschnitt gilt unverändert für `target_buffer_s=None`; mit
Regelung ersetzt „Scheibe 1b“ unten die feste Größe D durch den geregelten Puffer B.)

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
`rtt_exceeds_buffer` (Lag des Ziels + Reserve > D), `not_locked` (auch während des Neu-Einrastens, siehe unten), `estimate_stale`, `target_silent`, `too_late`
(Zeitpunkt liegt schon vor dem Ausgaberand), `out_of_order`, `already_active`, `unknown_source`. Ist die Schätzung
noch nicht eingerastet, prüft der Mixer gegen den RTT-Hinweis aus `register_source(name, rtt_hint_s=…)`.

**Neu-Einrasten nach jedem Wechsel der aktiven Quelle.** Jeder wartende DJ hört den Feed der aktiven Quelle; wechselt
sie, springt sein Lag um den Lag-Unterschied der beiden (`lag = lag_aktiv + RTT`). Die Nachführung sucht nur ±`gate_s`
(3 ms) um die Vorhersage und fände den neuen Lag nicht. Deshalb rastet jede wartende Quelle, die bisherige
Referenzquelle eingeschlossen, nach dem Wechsel neu ein: Messfenster erst ab `reacquire_delay_s` (3 s) danach,
gesucht wird ±`reacquire_window_s` (100 ms, weniger als ein halber Beat) um Lag(neu aktiv) + bisherige RTT der Quelle
(der Wechsel ändert die Bezugsquelle, nicht die Netzlaufzeit; ohne bekannte RTT oder nach `reacquire_narrow_s` = 8 s
ohne Erfolg im vollen Bereich), zwei stimmige Messungen; bis dahin gilt die alte Schätzung weiter (für Audio, das noch im Puffer
liegt), eine Übergabe an die Quelle wird mit `not_locked` verweigert, die Driftrate bleibt. Gemessen wird eine Quelle
erst, wenn die Ausgabe ihr letztes Segment hinter sich hat. Maxima am Rand des Suchbereichs werden verworfen.

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
Szenario A→B→A…: jeder DJ mischt nach der Übernahme zwei eigene Tracks hinein, der andere hört zu (Rückkanal =
Roh-Feed der abgebenden Quelle mit Downlink-Latenz, `mixer.read_feed`), startet seinen Track phasengenau mit menschlichem Fehler (Standard ±5 ms gleichverteilt) und
mischt über N Takte (Bass-Swap in der Mitte, Rückkanal per EQ/Fader ausgeblendet). Der Rückkanal ist per Tiefpass
in Bass/Höhen geteilt. Transport: feste Latenz je Richtung = RTT/2 einschließlich Jitter-Puffer, Pakete zu 128 Samples
mit Zufallsverlust und exponentialverteilter Verspätung (verspätet = verloren), Verlust = Nullen. Taktdrift: A läuft
+ppm, B −ppm schnell (Lag wandert mit der Zeit). Die erste Quelle definiert T; der Lag jedes späteren DJs ist
wahrer Lag der Programmquelle + eigene RTT + Drift (siehe Grenzen: Lag summiert sich). Die wahren Lags laufen im
Simulator getrennt von der Zeitachse des Mixers (die mit *geschätzten* Lags geführt wird); der aufsummierte
Schätzfehler (Versatz und Driftrate der Achse, `grid_offset_ms`, wenige ms) wird bei allen Messungen herausgerechnet
(`Frames` in `sim/scenario.py`), weil er im Raster steckt und nichts zwischen den Quellen verschiebt.

**Report** (`report.json`): `summary`, `acceptance`, `link_stats`, `mixer_events` und je Übergabe:

| Feld | Bedeutung |
|---|---|
| `rtt_true_ms`, `rtt_est_ms`, `rtt_error_ms` | wahre/geschätzte RTT (Lag relativ zur Programmquelle), Fehler = Schätzung − Wahrheit |
| `lag_true_ms`, `lag_est_ms`, `grid_offset_ms` | kumulierter Lag relativ zur Zeitachse des Mixers (wahr: im Raster des Mixers ausgedrückt); Versatz der Mixer-Achse gegen das wahre Raster |
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

# Scheibe 1b: elastischer Ausgang (Puffer-Regelung)

Problem aus Grenze 1: Jede Übergabe verschiebt den Live-Rand des Programms um die RTT des Übernehmenden nach
hinten (`lag_neu = lag_alt + RTT_neu`). Bei festem Ausgabepuffer D schrumpft der Rest `D − lag` mit jeder
Übergabe, bis keine mehr möglich ist. Statt D festzuhalten, wird der Puffer **geregelt**.

## Modell

* **Puffer** `B = Live-Rand − Ausgabeposition = D_eff − lag_aktiv`: Audio, das beim Server angekommen, aber noch
  nicht ausgegeben ist. `D_eff = w − T_aus` ist die gesamte Verzögerung des Ausgangs gegen die Wanduhr.
* **Jede Übergabe senkt B** um den Lag-Zuwachs (≈ RTT des Übernehmenden).
* **Das Ausgabe-Tempo r** ist die Rate, mit der der Ausgang die Zeitachse liest. Mit `r < 1` (etwas langsamer als
  Echtzeit) wächst B um `(1−r)` je Sekunde, mit `r > 1` schrumpft B. Bei `|r−1| ≤ 0,5 %` sind das höchstens
  5 ms Puffer je Sekunde, 0,3 s je Minute.
* **Nach einer Übergabe wird also LANGSAMER ausgegeben** (der Puffer füllt sich wieder auf), schneller nur, wenn
  B über dem Ziel liegt (zum Beispiel nach dem Start, wenn mit kleinem Anfangs-D begonnen wurde). Bei 174 BPM sind
  0,5 % = 173,1 BPM. Die Zuschauer-Verzögerung bleibt dauerhaft klein (rund B plus halbe RTT des live spielenden DJs),
  obwohl `D_eff` über die Sitzung um die Summe aller Lags wächst.
* **Die DJs hören davon nichts.** Ihr Rückkanal ist der Feed der abgebenden Quelle (`read_program(…, source=…)`),
  nicht der Ausgang; das Tempo betrifft nur, was an die Plattform geht. Die Phasenausrichtung der Übergaben
  läuft in Programmzeit T und ist vom Tempo unabhängig.
* **Tonhöhe:** Varispeed ohne Tonhöhenkorrektur. 0,5 % sind 8,6 Cent, also im Bereich normalen Pitch-Fader-Spiels.

## Regler (`control.py`)

`RelayConfig(target_buffer_s=3.0, min_buffer_s=1.0, max_dev=0.005, max_slew=0.001, deadband_s=0.1)`:

| Parameter | Standard | Bedeutung |
|---|---|---|
| `target_buffer_s` | `None` (aus) | Zielpuffer B_ziel; `None` = fester Puffer D wie in Scheibe 1, r ≡ 1. Empfehlung 3,0 s |
| `min_buffer_s` | 1,0 s | Untergrenze B_min: Übergabe nur, wenn B danach darüber bleibt |
| `max_dev` | 0,005 | größte Abweichung `|r−1|` (0,5 %) |
| `max_slew` | 0,001 /s | größte Änderung von r je Sekunde (weiche Rampen: von 1 auf 0,995 dauert 5 s) |
| `deadband_s` | 0,1 s | Totband ±: innerhalb bleibt r = 1 |
| `buffer_s` | 2,0 s | Anfangspuffer (der Mixer startet mit D = `buffer_s` und regelt dann auf das Ziel) |

Regelgesetz: Außerhalb des Totbands greift der Regler ein und führt B bis auf `deadband/2` ans Ziel (Hysterese,
keine Schwingung um den Zielwert). Der Sollwert für `r−1` folgt der Bremsparabel `sqrt(max_slew · Abweichung)`,
gedeckelt auf `max_dev`; r selbst darf sich nur mit `max_slew` ändern. Dadurch läuft r mit endlicher Steigung auf 1
aus, bevor B das Ziel erreicht (kein Überschwingen; Test).

## Ausgangsstufe (`varispeed.py`)

`VarispeedReader` liest das Programm P(T) mit fraktionaler Leseposition. Die Rate ändert sich innerhalb eines
Blocks linear, die Position ist die exakte Summe `p0 + r0·i + (r1−r0)·i²/(2n)`; damit gibt es keine Phasensprünge
an Blockgrenzen und keine Klicks bei Ratenwechsel. Position als `(int, Bruchteil)`, damit die Genauigkeit nicht
von der Laufzeit abhängt. Interpolation: Kaiser-Sinc mit 32 Stützstellen (Tabelle, 32768 Phasen) statt der
kubischen Interpolation von `timeline.py`, weil der Bruchteil beim Tempo 0,995 ständig wandert (alle 200 Samples
einmal durch) und kubisches Lagrange dabei die Höhen mit 240 Hz amplitudenmoduliert (−2 dB bei 14 kHz, −5 dB bei
18 kHz); der Sinc-Kern ist bis 20 kHz auf 0,01 dB flach und liefert bei Bruchteil 0 exakt den Eingangswert.

**Austauschbar:** `render(read, n, r0, r1)` bekommt eine Lesefunktion `read(t0, n) → P(t0 … t0+n)` und liefert n
Samples; `position`, `margin` beschreiben den Stand. Eine Keylock-/Time-Stretch-Stufe (z. B. librubberband, Rate als
Zeitverhältnis bei fester Tonhöhe) kann denselben Platz einnehmen, ohne dass Regler, Mixer oder Simulation
geändert werden.

## Übergabe-Freigabe

`request_handoff` / `check_handoff` prüfen mit Regelung statt `Lag + Reserve ≤ D`:

    Puffer jetzt − erwarteter Lag-Zuwachs ≥ min_buffer_s

Der Lag-Zuwachs ist `Lag(Ziel) − Lag(aktiv)` (Schätzung, sonst RTT-Hinweis). Verweigerung: Code `buffer_too_low`
mit Klartext (aktueller Puffer, Zuwachs, Rest, Untergrenze) und geschätzter Wartezeit bis zur Freigabe
(`HandoffResult.wait_s`, auch im Text: „voraussichtlich frei in 26 s“). Die Wartezeit rechnet das Regelgesetz
vorwärts durch (`BufferController.time_to`); der Test prüft sie gegen den tatsächlichen Zeitpunkt (±20 %).
„Nicht erreichbar“ meldet sie, wenn die nötige Pufferhöhe hinter dem Ruhepunkt des Reglers liegt.
Alle anderen Gründe (`not_locked`, `estimate_stale`, …) bleiben wie in Scheibe 1.

## Status-API (Vorbereitung für die Oberfläche)

`mixer.status.buffer` (`BufferStatus`):

| Feld | Bedeutung |
|---|---|
| `buffer_s` | aktueller Puffer B (Live-Rand aus dem Lag-Modell minus Ausgabeposition; die Regelgröße) |
| `supply_s` | echter Vorrat: Ende des Rings der aktiven Quelle (`ring.end`, was wirklich angekommen ist) minus Leseposition der Ausgabe in diesem Ring. Gegenprobe zu `buffer_s`; beide sind gleich, solange Ring-Index und Wanduhr übereinstimmen. Negativ = die Ausgabe liest Nicht-Angekommenes (stummes Programm, `underruns` steigt). `None` ohne aktive Quelle |
| `rate`, `rate_bpm` | Ausgabe-Tempo r und äquivalente BPM (`bpm_ref` · r, Standard 174, z. B. 173,1) |
| `release`, `wait_s` | Freigabe der Übergabe nach Pufferkriterium (ja/nein) und Sekunden bis dahin (0 = frei) |
| `handoff[quelle]` | je Ziel: `ok`, `buffer_after_s`, `lag_growth_s`, `wait_s` |
| `eta_target_s` | Sekunden, bis B im Ruhebereich um das Ziel liegt |
| `output_delay_s`, `lag_s` | `D_eff = w − T_aus` und Lag der aktiven Quelle |

Für Traces gibt es `mixer.last_block` (Leseposition, Rate am Blockanfang/-ende, Puffer), `mixer.read_position()`
(Lag, Leseposition und Ringende der aktiven Quelle als Wanduhr-Index) sowie `mixer.buffer_s`, `mixer.supply_s`,
`mixer.rate`, `mixer.output_position`. `mixer.set_lag(name, lag_s)` setzt einen von außen gemessenen Lag (ohne
Audio-Schätzung, z. B. für Tests).

## Zeitachse und Speicher über Stunden

Der absolute Lag und `D_eff` wachsen über die Sitzung (30 Übergaben bei 40/250 ms: +4,35 s Lag). Nichts im
Mixer hängt am absoluten Lag:

* Ringe halten nur das Fenster Ausgabe…Ankunft (B plus RTT der Übergabe) und das Schätzfenster:
  `ring_s = est_window + search_max + max(buffer_s, target) + deadband + 2` (ca. 12 s je Quelle).
* Der Index im Ring einer Quelle ist die Wanduhr: eine Quelle, die erst während der Sitzung angemeldet wird, beginnt
  bei der Wanduhr ihrer Anmeldung (`SampleRing(capacity, start)`). Bis S3904 begann jeder Ring bei 0; ein DJ, der
  sich mehr als eine Pufferlänge nach dem Start verband, wurde um seine Anmeldezeit zu spät gelesen (stummes Programm,
  siehe „Befund S3904“).
* Die RTT-Suche läuft relativ zum Lag der aktiven Quelle (`search_max_s` = größte gesuchte RTT darüber, Standard 3 s),
  nicht mehr absolut.
* Ausgabeposition als `(int, Bruchteil)`; Segmente werden relativ zur Ausgabeposition verworfen; `events` ist ein
  begrenzter Ring (4096).
* Test über 3 h simulierte Zeit (2 kHz Abtastrate, Übergabe alle 60 s, 179 Übergaben, Lag und `D_eff` > 25 s): Ringe
  und Listen bleiben gleich groß, Prozessspeicher wächst nicht, die Ausgabe liegt am Ende noch auf < 2·10⁻⁴ an
  der Sinuskurve der gemeldeten Leseposition.

## Simulation mit Regelung

```bash
python -m b2b_relay.sim --handoffs 30 --rtt-a-ms 40 --rtt-b-ms 250 --jitter-ms 15 --loss 0.001 \
       --drift-ppm 50 --seed 1 --target-buffer-s 3 --max-dev 0.005 --out /tmp/r.json
```

Neue Schalter: `--target-buffer-s` (3,0; `0` = fester Puffer wie in Scheibe 1), `--min-buffer-s` (1,0), `--max-dev`
(0,005), `--max-slew` (0,001), `--deadband-s` (0,1), `--min-gap-s` (60: Mindestabstand zwischen zwei Übergaben,
verlängert die Haltezeit nach den eigenen Übergängen; Standard-Drehbuch: je DJ zwei Übergänge, dann Übergabe),
`--settle-s` (90: Nachlauf nach dem Programmende, damit der Puffer ausregeln kann). `--buffer-s` ist der
Anfangspuffer (2,0). Die Messungen laufen weiter in Programmzeit: der Ausgang wird dafür über die aufgezeichnete
Leseposition zurück auf das T-Raster interpoliert (`OutputMap`).

Zusätzlich im Report (`report.json`):

| Feld | Bedeutung |
|---|---|
| `buffer` | `min_s`, `max_s`, `start_s`, `final_s`, `target_s`, `min_limit_s`, `output_delay_final_s` |
| `rate` | `min`, `max`, `final`, `max_abs_dev`, `bpm_min`, `bpm_max`, `bpm_final` |
| `viewer_delay` | Zuschauer-Verzögerung relativ zum live spielenden DJ: `max_s`, `min_s`, `mean_s`, `final_s` |
| `series` | alle 10 s: `t_s`, `buffer_s`, `rate`, `bpm_equiv`, `viewer_delay_s`, `dj` |
| je Übergabe | `buffer_at_request_s`, `buffer_after_expected_s`, `buffer_min_around_s`, `rate_at_handoff` plus die bisherigen Kennzahlen |
| `acceptance` | zusätzlich `buffer_never_below_min`, `viewer_delay_below_4s`, `rate_within_max_dev`, `buffer_ends_near_target` |

Zuschauer-Verzögerung = Wanduhr − Zeitpunkt, zu dem der jeweils live spielende DJ den gerade ausgegebenen Inhalt
gespielt hat (Ankunft beim Server − RTT/2, aus der wahren Lag-Funktion des Simulators).

### Ergebnis des Abnahmelaufs (Seed 1, 30 Übergaben, 31 min Programm)

| Kennzahl | Wert |
|---|---|
| Übergaben akzeptiert | 30/30 (Abstand 62 s) |
| Puffer B | min 1,99 s (Start), max 2,95 s, Ende 2,95 s; nach den Übergaben nie unter 2,27 s; Untergrenze 1,0 s |
| Tempo r | 0,9950 … 1,0000 (173,13 … 174,00 BPM), `max |r−1|` = 0,0050 |
| Zuschauer-Verzögerung | max 3,20 s, Mittel 2,94 s, Ende 3,10 s (Grenze 4 s) |
| Ausgabeverzögerung `D_eff` am Ende | 7,44 s (Summe der Lags 4,46 s) |
| RTT-Fehler | max 0,40 ms |
| Beat-Phase an Übergaben | max 1,26 ms (Grenze 3 ms) |
| Lücken (unerklärt) | 0 |
| Laufzeit / Speicher | 2:28 min, RSS 1,55 GB |

## Grenzen (Scheibe 1)

* **Der Lag summiert sich über Übergaben.** Der übernehmende DJ hört das Programm erst, wenn es beim Server
  angekommen ist, und schickt es um seine RTT versetzt zurück: `lag_neu = lag_alt + RTT_neu`. Bei festem
  Puffer D gilt die Prüfung „D > RTT + Reserve“ für den *kumulierten* Lag: Bei 40/250 ms RTT wächst er in
  zehn Übergaben auf rund 1,45 s; bei D = 2 s und 300 ms Reserve ist dann Schluss. **Scheibe 1b löst das
  mit dem elastischen Ausgang** (nächster Abschnitt); ohne Regelung (`target_buffer_s=None`) bleibt die Grenze.
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

## Grenzen (Scheibe 1b)

* **Der Regelbereich ist endlich.** Der Puffer kann höchstens um `max_dev` (5 ms/s) wachsen. Dauerhaft
  tragbar ist eine mittlere Lag-Zunahme von weniger als 5 ms je Sekunde, also zum Beispiel eine Übergabe mit
  250 ms alle 50 s oder mit 120 ms alle 25 s. Schnellere Folgen verweigert die Freigabe (`buffer_too_low`) mit
  Wartezeit, statt den Puffer zu unterschreiten.
* **Wartezeit** nimmt an, dass bis dahin keine weitere Übergabe und keine Änderung der Reglerwerte kommt.
* **Anfangsphase.** Mit `buffer_s` < Ziel steigt die Ausgabeverzögerung nach dem Start, bis B das Ziel erreicht
  (bei 2,0 → 3,0 s etwa 200 s lang Tempo 0,995).
* Varispeed ohne Keylock: Tonhöhe und Tempo ändern sich gemeinsam (≤ 0,5 %). Wer Keylock will, setzt eine
  Time-Stretch-Stufe an dieselbe Schnittstelle.
* Die Zuschauer-Verzögerung enthält die halbe RTT des live spielenden DJs; sie wird in der Simulation aus der
  wahren Lag-Funktion berechnet, im Betrieb nur aus der Schätzung.
* Die Simulation misst im T-Raster; dafür wird der Ausgang kubisch zurückinterpoliert (nur Messpfad, nicht
  Produktpfad).
* **Die Zeitachse des Mixers driftet gegen das wahre Raster.** Der Mixer misst jeden neuen Lag relativ zum
  Programm, das er mit dem *geschätzten* Lag der vorigen Quelle zusammensetzt; Fehler der Driftrate (einige 10 ppm
  je Übergabe) summieren sich deshalb über die Kette (Simulation: `grid_offset_ms` bis 118 ms nach 30 Übergaben,
  im Mittel ≈ 60 ppm Tempoabweichung der Achse). Zwischen den Quellen verschiebt das nichts (RTT-Fehler und
  Beat-Phase bleiben klein), das Programm läuft aber mit leicht falscher Taktrate. Vorschlag in der PR-Beschreibung.

---

# Scheibe 2a: Live-Server (Betrieb)

Der `RelayMixer` läuft als JACK-Client `b2b-relay` hinter einem JackTrip-Hub-Server. Alles in einem Container:
`jackd` (Dummy-Treiber, 48 kHz, Periode 256) → `jacktrip -S -p 5` (Hub-Server, **ohne Auto-Patching**) → `python -m b2b_relay.live`.

```
DJ-Rechner --jacktrip--> [Hub-Worker "alice": alice:receive_N] --JACK--> b2b-relay:in_alice_L/R
                                                                              | Ring (lock-frei, kein Alloc)
                                                                              v
                                              Worker-Thread: RelayMixer (Mitte) + Folger (Seite), RTT-Schätzung, Drift
                                                                              | Ring
                                                                              v
                                   b2b-relay:program_L/R --JACK--> Hörer-Clients (<name>:send_N), optional WAV
Rückkanal (ohne Python):  <live>:receive_N --JACK--> <wartender DJ>:send_N   (direkte Portverbindung)
```

## Start

```bash
docker compose up -d                       # Image ghcr.io/kanne87/b2b-relay:latest, network_mode: host
curl -s localhost:8090/status | jq .       # Vorgabe: API nur auf 127.0.0.1
```

`docker compose build` baut lokal: JackTrip v2.7.2 aus dem Quelltext **ohne GUI** (`-Dnogui=true -Dnovs=true`),
Ubuntu 24.04, Python-Paket mit Extra `[live]`. Ohne Docker: `pip install -e .[live]`, `jackd -d dummy -r 48000 -p 256`,
`jacktrip -S -p 5 -q auto`, `python -m b2b_relay.live`.

## Ports (network_mode: host, kein NAT-Mapping)

| Port | Protokoll | Zweck |
|---|---|---|
| 4464 | TCP | JackTrip-Hub-Handshake (Clients verbinden sich hierhin) |
| 61002 – 62000 | UDP | Audio der Hub-Worker, je verbundenem Client ein Port ab `JACKTRIP_UDP_BASE` (bei 3–5 DJs 61002–61006) |
| 8090 | TCP | HTTP-API (`B2B_API_PORT`), Vorgabe nur `127.0.0.1` |

## Umgebung

| Variable | Vorgabe | Bedeutung |
|---|---|---|
| `JACK_RATE` / `JACK_PERIOD` | 48000 / 256 | Samplerate und Periode des Dummy-Treibers (Samplerate fest 48 kHz) |
| `JACK_REALTIME` | 0 | 1 = `jackd -R` (braucht `cap_add: SYS_NICE`, `ulimits: rtprio 95, memlock -1`) |
| `JACKTRIP_HUBPATCH` | 5 | Patch-Modus des Hubs; 5 = kein Auto-Patching (nicht ändern) |
| `JACKTRIP_QUEUE` / `JACKTRIP_UDP_BASE` / `JACKTRIP_EXTRA` | auto / 61002 / – | Queue des Servers, UDP-Basisport, weitere Schalter |
| `B2B_API_HOST` / `B2B_API_PORT` | 127.0.0.1 / 8090 | Bindung der API. Für Zugriff von außen `0.0.0.0` **und** `B2B_API_TOKEN` setzen |
| `B2B_API_TOKEN` | leer | Bearer-Token; leer = keine Prüfung |
| `B2B_CHUNK` | 2048 | Frames je Worker-Block (Vielfaches von 64) |
| `B2B_TARGET_BUFFER_S` / `B2B_MIN_BUFFER_S` / `B2B_MAX_DEV` | 3.0 / 1.0 / 0.005 | Regelung des elastischen Ausgangs (Scheibe 1b) |
| `B2B_DEFAULT_ROLE` / `B2B_ROLES` | dj / – | Rolle neuer Clients; `B2B_ROLES="tv=listener,kai=dj"` legt Rollen nach Name fest |
| `B2B_AUTO_ACTIVE` | true | Der erste verbundene DJ ist live (sonst setzt `POST /handover` den ersten) |
| `B2B_RECORD_DIR` / `B2B_RECORD_FORMAT` | – / pcm16 | Mitschnitt der Programmausgabe als WAV (`pcm16` oder `float32`), daneben `*.trace.jsonl` |
| `B2B_DRIFT_ENABLE` / `B2B_DRIFT_DEADBAND_MS` / `B2B_DRIFT_SLEW_MS_PER_S` | true / 1 / 1 | Verankerung der Zeitachse an der Wanduhr |
| `B2B_IGNORE_CLIENTS` | system,JackTrip,… | JACK-Clients, die nie als DJ gelten |
| `B2B_MAX_RATE_PPM` | 0 | Driftrate der Lag-Schätzung (0: hinter dem Hub gibt es keine, siehe „Lag-Schätzung im Betrieb“) |
| `B2B_CONCEAL_MAX_MS` | 100 | Aussetzer (exakte Nullen) im Feed bis zu dieser Länge verschleiern |
| `B2B_DEBUG_POS` | false | Diagnose: je Sekunde eine Zeile `b2b-debug {…}` auf stderr (Container-Log) mit Lag, Leseposition und Ringende der aktiven Quelle (`pos`), `buffer_s`, `supply_s`, `program_level_db`, `underruns` |

**Empfehlung `JACKTRIP_QUEUE=12`** (statt `auto`): siehe „Lag-Schätzung im Betrieb“. Der Integrationstest läuft mit 12.

## Client-Aufruf für DJs

```bash
jacktrip -C <host> --remotename <name> -n 2 -q auto
```

Der Name ist die Identität im Status, in `/roles` und `/handover`. Ohne `--remotename` (in der GUI: kein
„Remote Name“) benennt der Hub den Client nach seiner Adresse, z. B. `__ffff_<ip>`; er funktioniert als DJ genauso
(Rolle aus `B2B_DEFAULT_ROLE`, `/handover` mit diesem Namen), nur ist der Name nicht sprechend und wechselt mit der IP. Linux mit JACK: wie oben, Audio in/aus
`JackTrip:send_*` / `JackTrip:receive_*`. **macOS/Windows ohne JACK** nutzen das RtAudio-Backend (in den
JackTrip-Paketen enthalten): `jacktrip -R -C <host> --remotename <name> -n 2 -q auto` (Gerät wählen mit
`--audiodevice`, Liste mit `--listdevices`). Feste Samplerate 48 kHz.

## Routing und Rollen

* Jeder Client mit `receive_N`-Ports erscheint im Status. Rolle **dj**: sein Audio geht in den Mixer. Rolle
  **listener**: er bekommt `program_L/R` zurück und liefert nichts.
* **Rückkanal:** Jeder DJ, der nicht live ist, bekommt den Roh-Feed der live spielenden Quelle über eine direkte
  JACK-Verbindung (`<live>:receive_N → <dj>:send_N`). Bei jeder Übergabe (Wechsel von `live`) schaltet der Router
  um (Zeitraster `B2B_ROUTER_INTERVAL_S`, 0,5 s). Der Live-DJ hört nichts zurück. Kein Python, kein zusätzlicher Puffer.
* Der Mixer ist einkanalig; Stereo läuft als Mitte (RTT-Messung, Entscheidungen) und Seite (Folger mit denselben Lags
  und Umschaltpunkten), Ausgabe `L = M+S`, `R = M−S`.

## HTTP-API (FastAPI, keine Oberfläche)

```bash
curl -s localhost:8090/status
curl -s -XPOST localhost:8090/handover -H 'content-type: application/json' -d '{"to":"bob"}'
curl -s -XPOST localhost:8090/roles    -H 'content-type: application/json' -d '{"client":"tv","role":"listener"}'
```

* `GET /status`: `live`, `previous`, `buffer` (Puffer B, Rate, äquivalente BPM, Freigabe/Wartezeit der Übergabe je Ziel),
  `sources` (je Quelle `rtt_ms`, `jitter_ms` = Streuung der Lag-Messungen, `drift_ppm`, Konfidenz, Pegel), `drift`,
  `jack` (`xruns`, `out_underruns`, `cb_errors`, `cpu_load`), `worker` (Rechenlast, verworfene Frames), `clients`
  (Name, Rolle, `live`, was der Client hört), `events`, `recording`. `GET /healthz` für Healthchecks.
  Ob das Programm wirklich Ton führt: `program_level_db` (RMS der Programmausgabe über die letzte Sekunde, dBFS,
  beide Kanäle; −120 = digitale Stille), `program_silent_s` (Sekunden, seit denen jeder Ausgabeblock unter −60 dBFS
  liegt; 0 = Ton) und `buffer.supply_s` (echter Vorrat der aktiven Quelle, siehe „Status-API“; `buffer.buffer_s` ist
  die Regelgröße aus dem Lag-Modell). `underruns` zählt Ausgabeblöcke, deren Leseposition hinter dem Ende des
  Quellrings liegt; im eingeschwungenen Betrieb steigt er nicht.
* `POST /handover {"to": name}`: 200 bei Annahme; 409 mit `reason`/`message`/`wait_s` bei Verweigerung
  (z. B. `buffer_too_low`, `not_locked`), 404 bei unbekanntem Namen. Ohne aktive Quelle setzt es die erste.
* `POST /roles {"client", "role": "dj"|"listener"}`: der live spielende DJ kann nicht Hörer werden (409).

## Zeitachse und Wanduhr (Drift)

Die Programmzeitachse T wird an `CLOCK_MONOTONIC` verankert (`live/drift.py`). Gemessen wird die Abweichung des Live-Rands
von der Sollkurve `T_ref + Δt_wall − Σ Lag-Zuwachs der Übergaben` (Achsendrift aus Grenze 1b plus Lauf der JACK-Uhr gegen
die Wanduhr). Der Rest wird als **Umlabeln** der Achse ausgeregelt (`RelayMixer.retime`, ≤ 1 ms/s, Totband 1 ms): das
ändert kein Audio und keine Phase zwischen den Quellen. Die Varispeed-Stufe regelt weiter den Puffer B (sie kann
die Lage der Achse nicht verändern, nur die Ausgabeposition gegen den Live-Rand); die Wahl „Umlabeln statt Tempo“
ist im PR begründet. Im Status unter `drift`: `drift_ms` (Regelgröße, Ziel < 10 ms), `raw_drift_ms` (ohne Eingriff),
`corrected_ms`, `ppm_60s`, `max_abs_ms`. Alle Lags tragen einen Vorlauf von `B2B_DRIFT_BASE_S` (10 s), damit sie
positiv bleiben; im Status stehen nur RTTs (relative Lags).

## Lag-Schätzung im Betrieb (hinter dem JackTrip-Hub, S3904)

Hinter dem Hub läuft jeder Feed im Takt des Servers: der Jitterpuffer von JackTrip (Strategie 1) gleicht Uhren- und
Netzschwankungen aus, indem er ganze Slots (ein Paket = 256 Frames = 5,3 ms) einfügt (Unterlauf: Nullen, Lag +1 Slot)
oder verwirft (Überlauf: Lag −1 Slot); Verlust ergibt Nullen. Der Lag einer Quelle ist also stückweise konstant und
springt in Slots, eine stetige Driftrate gibt es nicht. Der Live-Server (`live/engine.py`) stellt den Mixer darauf ein:

| Einstellung | Wert | Wirkung |
|---|---|---|
| `max_rate_ppm` | 0 | keine Driftrate. Vorher lief eine falsch geschätzte Rate (bis 500 ppm) bei jeder Übergabe auf alle Quellen über und verschob die Phase um ≈ Rate · 30 s |
| `confirm_s` | 4,5 s | Einrasten nur aus zwei Messungen getrennter Fenster: ein einzelnes Ereignis (Knacks, Aussetzer, Signalbeginn) kann nicht beide tragen |
| `verify_every_s`, `verify_ratio`, `verify_strikes` | 2 s, 0,5, 3 | Prüfung gegen den vollen Suchbereich: erreicht der Peak an der Schätzung dreimal in Folge nicht die Hälfte des größten Peaks, wird die Einrastung verworfen (Ereignis `lock_rejected`) und neu gesucht |
| `track_window_s`, `accept_s`, `fit_span_s` | ±30 ms, ±12 ms, 12 s | Nachführung: Messungen bis ±24 ms gehen in den Verlauf; geschätzt wird der Mittelwert der Messungen ±12 ms um deren Median über 12 s (der mittlere Versatz der Musik, wenn der Feed zwischen Slot-Zuständen wechselt); weiter weg = Ausreißer, nach 2 s stimmiger Ausreißer (`relock_s`) neu aufgesetzt |
| `reacquire_narrow_s` | 12 s | Neu-Einrasten nach einem Wechsel zuerst nur um Lag(aktiv) + bisherige RTT |
| `conceal_max_s` | 0,1 s | `conceal.py`: Läufe exakter Nullen (≥ 16 Samples, ≤ 100 ms) zwischen Signal werden im Quellring durch das Signal davor ersetzt (Wellenform-Ersatz, Kosinus-Überblendung 64 Samples). Das geht nachträglich, weil der Mixer jede Quelle erst rund eine Pufferlänge später ausgibt. Längere Stille bleibt (Fallback) |

Die Vorgaben von `RelayConfig` lassen alles beim Verhalten aus Scheibe 1 (Simulation mit stetiger Drift).

**JackTrip-Queue.** Mit `-q auto` hält der Hub nur 4–5 Slots (wan40) bzw. 9–15 (wan120) vor (Server-Log
„AutoQueue“) - gemacht für Musiker, die wenig Latenz brauchen. Bei 5 ms Jitter läuft der Puffer dann rund alle 2 s
leer: der Feed des live spielenden DJs liegt in der Nachbildung 29 % der Zeit einen Slot später, und die Messung des
Phasensprungs rauscht schon ohne Übergabe bis 1,1 ms. Das Relay hat 3 s Puffer und braucht keine kurze Queue: mit
`JACKTRIP_QUEUE=12` (64 ms, die RTT wächst um rund 40 ms) verschiebt die Strecke bei wan40 nichts mehr. Bei Verlust
und Umordnung (wan120) bleiben Slot-Sprünge (Nullmessung bis 2,4 ms mit 12, bis 6,7 ms mit `auto`).

## Befund S3904: stummes Programm bei spät verbundenem DJ

Live-Probe 13.11.2026 (ein Windows-Client, RtAudio, ohne Remote Name): Programm 100 % exakte Nullen, `underruns`
+23,5/s (jeder Worker-Block), `buffer_s` 3,0, Pegel der Quelle −23…−32 dB. Ursache: Der Mixer legte den Ring einer
Quelle bei ihrer Anmeldung mit Index 0 an, liest ihn aber nach Wanduhr (Index = Samples seit Start der Engine). Der
DJ verband sich etliche Sekunden nach dem Start; seine Daten lagen im Ring um die Anmeldezeit `W` zu früh, die
Leseposition (Wanduhr − 3 s) also `W − 3 s` hinter `ring.end` - Nullen und ein Unterlauf je Block. `buffer_s` kommt aus
dem Lag-Modell und merkt davon nichts. Im Integrationstest verbinden sich die Test-DJs, während der Server startet
(CI-Run 15: alle Container in 0,2 s gestartet, alle DJs nach 6,5 s da, angemeldet vor oder kurz nach dem Start der
Engine): Versatz unter der Pufferlänge, das Programm war hörbar. Fix: Ring beginnt bei der Wanduhr der Anmeldung.
Regression: `tests/test_live_late_join.py`; der Integrationstest prüft den Programmpegel je Haltezeit und fährt das
Profil `noname` (später Join ohne Remote Name).

## Callback und xruns

Der JACK-Callback kopiert nur zwischen Port-Puffern und `AudioRing` (lock-frei, ein Schreiber/ein Leser, Zähler statt
Lock, vorab angelegter Speicher; `tests/test_live_ring.py` misst per `tracemalloc`, dass kein Datenpuffer entsteht) und
fängt jede Ausnahme (`cb_errors`). `RelayMixer.process` und die RTT-Schätzung laufen im Worker-Thread mit Blöcken
von 2048 Frames; der Programmausgang hat ohnehin 3 s Puffer. Der Callback ist Python und braucht das GIL
(`sys.setswitchinterval(0.0005)`); xruns zählt der JACK-Callback (`jack.xruns`). Kein Echtzeit-Scheduling im Container
voreingestellt (`JACK_REALTIME`).

## Integrationstest (Docker, tc netem)

```bash
python tests/integration/run.py --dry-run                    # Ablauf anzeigen, startet nichts
python tests/integration/run.py                              # 3 Profile x 10 Übergaben + noname, ca. 43 min + Image-Bau
python tests/integration/run.py --profiles clean --handovers 3 --hold-s 45 --out /tmp/itest
```

`docker-compose.yml` in `tests/integration/`: Server + 3 Test-DJs (`client.py`: eigenes `jackd` dummy, `jacktrip`-Client,
Kicks bei 174 BPM mit je eigener Klangfarbe, leise Hi-Hats auf pseudozufälligen Sechzehnteln (je DJ eigenes Muster) plus
leises Pad; der Übernehmende rastet auf das Raster des gehörten Feeds ein, reicht den Feed durch und mischt seine Kicks
phasenrichtig dazu, wie in `sim/`). Ohne die Hats wäre das Testsignal exakt periodisch und die Korrelation nur an
Einzelereignissen eindeutig - genau dort entstanden die falschen Einrastungen (S3904). `tc netem` läuft im
Client-Container (`cap_add: NET_ADMIN`) auf dessen Upload; der Server bleibt unverändert, bis auf die JackTrip-Queue
des Hubs: fest 12 Slots (`--jacktrip-queue`, Vorgabe 12; `auto` wie bisher möglich). Profile:

| Profil | netem (Client-Egress) |
|---|---|
| `clean` | keine |
| `wan40` | `delay 40ms 5ms distribution normal rate 1000mbit` |
| `wan120` | `delay 120ms 20ms distribution normal loss 0.5% reorder 5% 50% rate 1000mbit` |
| `noname` | keine; nur Server und ein Client **ohne `--remotename`** (`clientN`, Compose-Profil `noname`), der `--late-join-s` (20 s) nach dem Server startet. Der Orchestrator findet seinen Namen (`__ffff_<ip>` o. ä.) über `/status`, setzt ihn live und hält `--noname-hold-s` (90 s); keine Übergabe (S3904) |

`rate 1000mbit` hält die Paketreihenfolge: ohne `rate` sortiert netem jedes Paket nach seiner eigenen Zufallsverzögerung,
bei 5 ms Jitter und 5,33 ms Paketabstand kämen rund 23 % der JackTrip-Pakete hinter ihrem Nachfolger an und gingen als
Lücke verloren. Gewolltes Umordnen steht ausdrücklich im Profil (`reorder` in `wan120`).

Bricht ein Profil ab (Zeitüberschreitung, beendeter Container), gibt `run.py` die letzten 200 Zeilen der Compose-Logs
je Container aus (auch unter `<profil>/logs/`), wertet den Mitschnitt aus, soweit es ihn gibt, und schreibt
`analysis.json` mit Feld `error`; die übrigen Profile laufen weiter. Ohne Docker bildet `tests/itest_sim.py` dieselbe
Sitzung nach (Live-Server auf dem FakeJack-Prüfstand, drei `DjCore` aus `client.py` mit eigener RTT;
`tests/test_itest_sim.py`). Mit `tests/netsim.py` auch die Netzstörung: netem (`rate`-Zweig: Reihenfolge bleibt,
Verlust, Umordnung mit Korrelation) und der Empfang am Hub nach JackTrip v2.7.2 (`UdpDataProtocol`: ältere Pakete
verworfen, Lücken als Verlust; `JitterBuffer` Strategie 1 samt Auto-Queue). Die Nachbildung trifft die Werte aus CI-Run
13: AutoQueue 3 / 4–6 / 9–16, Aussetzer von 5,3 ms (wan40) bis 32 ms (wan120). `tests/test_relay_net.py` fährt damit
wan40 und wan120 (Regression S3904).

Ausgabe in `--out`: `report.md`, `results.json` und je Profil `rec/program-*.wav` (Mitschnitt der Programmausgabe),
`rec/*.trace.jsonl` (Block-Zeilen: Mitschnitt-Index, Inhaltszeit, Rate, Puffer, Drift, xruns; Ereignisse), `status.jsonl`,
`analysis.json` (je Übergabe auch `lag_s` des Ziels). Gemessen werden je Übergabe der Beat-Phasenfehler (Raster 22 s vor gegen 18 s nach dem Umschaltpunkt,
in Inhaltszeit, also unabhängig vom Varispeed), Lücken, Sprung-Verhältnis (Klicks) sowie xruns (Abnahme: nur während des Mitschnitts; die Startphase von jackd liegt davor) und Puffer-/Raten-Verlauf.
Dazu `null_jumps`: dieselbe Phasenmessung mitten in einer Haltezeit, ohne Übergabe (Grundrauschen unter der
Netzstörung), und `concealed`: Aussetzer der Strecke, die der Server verschleiert hat. Die Onset-Erkennung beginnt
120 ms vor jedem Messfenster; vorher zählte ein Fenster, das mitten in einem Kick begann, dessen Rest als Onset
(bis 1,06 ms Fehler ohne jede Übergabe). Grenzen für die Abnahme: `tests/integration/analyze.py` (`LIMITS`):
Sprung clean und wan40 ≤ 1 ms, wan120 ≤ 10 ms, keine Lücken in allen Profilen. Seit S3904 außerdem in allen
Profilen `program_audible` (`holds`: Pegel je Haltezeit ab 5 s nach dem Hörbarwerden bzw. nach jedem Umschaltpunkt
bis 1 s vor dem nächsten; kein 1-s-Block unter −60 dBFS) und `underruns_steady` (`underruns` aus `status.jsonl` steigt
in den letzten 30 s nicht); im Profil `noname` zusätzlich `client_without_remotename` (der Name ist keiner von A/B/C). In GitHub Actions läuft der Test im Job
`integration`, der Bericht ist Artifact und Job-Zusammenfassung.

## CI

`.github/workflows/ci.yml`: `unit` (pytest), `image` (baut; Push von `:latest` und `:<sha>` nach `ghcr.io/kanne87/b2b-relay`
nur bei Push auf `main`), `integration` (Docker + netem auf dem Runner, derzeit `continue-on-error`, bis der erste
grüne Lauf vorliegt).

## Lizenz

JackTrip-Kern: MIT (Expat). Die grafische Oberfläche („Classic mode“, aus QJackTrip) steht unter GPL-3. Der Dockerfile
baut JackTrip ausschließlich als Kommandozeilenprogramm (`-Dnogui=true -Dnovs=true`), es wird kein GUI-Code
übernommen oder ausgeliefert. Wer ein Binär-Paket von jacktrip.org für DJ-Rechner verteilt, verteilt dort das GUI mit.

## Grenzen (Scheibe 2a)

* Keine Authentifizierung der JackTrip-Verbindung (der Hub nimmt jeden Client an); die API ist nur per Token geschützt.
* Der Mixer rechnet mono (Mitte/Seite); ein Quellsignal in reiner Gegenphase wird nicht gemessen.
* Der Docker-Build und der Integrationstest sind in der Entwicklungsumgebung nicht ausgeführt worden (kein Docker; das
  Profil `noname` wird erst im CI-Lauf abgenommen); die
  Live-Pfade sind mit einem JACK-Ersatz (`tests/fakejack.py`) getestet, nicht gegen echtes `jackd`/`jacktrip`.
* Die Netzstörung wirkt nur auf den Upload der Clients. Die DJs hören den Rückkanal roh vom Hub: Aussetzer dort
  verschleiert der Server nicht (er sieht sie erst im zurückkommenden Feed).
* Slot-Sprünge im Feed des live spielenden DJs (JackTrip-Jitterpuffer bei Verlust/Umordnung) kann das Relay nicht
  sehen; sie verschieben das Programm um 5,3 ms für Bruchteile von Sekunden. Mit `JACKTRIP_QUEUE=auto` kommen
  Unterläufe schon bei 5 ms Jitter hinzu.
* Verschleierung ist Wellenform-Ersatz, kein Rekonstruieren: das Programm hat keine Stille, aber an der Stelle
  wiederholtes Audio (5–30 ms).
* Nicht Teil: Beat-Analyse, Weboberfläche, Phasemeter (Scheibe 3), Video (4), Begleit-App/Installer (5), Deploy.
