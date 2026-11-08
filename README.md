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

1. Kern: Relay-Mixer-Algorithmus (Zeitachse, RTT-Messung, Ausrichtung, Umschaltung) mit deterministischer Offline-Simulation.
2. Live-Audio: Transport (JackTrip-Hub/JACK) auf dem Server, Test mit künstlicher Netzverzögerung (tc netem).
3. Beat-Analyse + WebSocket + Weboberfläche (BPM, Takt/Phrase latenzkorrigiert, Live-Anzeige, Übergabe-Knopf, Leitungsqualität).
4. Video/OBS, Overlay, Ausspielung Twitch (DJ Program).
