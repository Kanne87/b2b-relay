"""Konfiguration des Live-Servers aus Umgebungsvariablen (alle mit Vorgabe)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Mapping


def _b(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "ja", "on")


@dataclass
class LiveConfig:
    # JACK
    client_name: str = "b2b-relay"        # B2B_CLIENT_NAME
    sample_rate: int = 48000              # B2B_SAMPLE_RATE (muss zum JACK-Server passen)
    chunk: int = 2048                     # B2B_CHUNK: Frames je Worker-Block (Mixer.process)
    # Mixer
    target_buffer_s: float = 3.0          # B2B_TARGET_BUFFER_S
    min_buffer_s: float = 1.0             # B2B_MIN_BUFFER_S
    max_dev: float = 0.005                # B2B_MAX_DEV
    fade_s: float = 0.1                   # B2B_FADE_S
    bpm_ref: float = 174.0                # B2B_BPM_REF (nur Anzeige)
    max_rate_ppm: float = 0.0             # B2B_MAX_RATE_PPM: Driftrate der Lag-Schätzung (0: hinter dem JackTrip-Hub
                                          # läuft jeder Feed im Takt des Servers, Uhrenfehler werden zu Slot-Sprüngen)
    conceal_max_ms: float = 100.0         # B2B_CONCEAL_MAX_MS: Aussetzer (exakte Nullen) bis zu dieser Länge verschleiern
    # Steuerung
    api_host: str = "127.0.0.1"           # B2B_API_HOST (im Container/Compose 0.0.0.0 + Token)
    api_port: int = 8090                  # B2B_API_PORT
    api_token: str = ""                   # B2B_API_TOKEN (leer = keine Prüfung)
    default_role: str = "dj"              # B2B_DEFAULT_ROLE: dj | listener
    roles: dict[str, str] = field(default_factory=dict)   # B2B_ROLES="alice=dj,tv=listener"
    auto_active: bool = True              # B2B_AUTO_ACTIVE: erster DJ wird live
    ignore_clients: tuple[str, ...] = ("system", "JackTrip", "PulseAudio JACK Sink")  # B2B_IGNORE_CLIENTS
    router_interval_s: float = 0.5        # B2B_ROUTER_INTERVAL_S
    # Mitschnitt
    record_dir: str = ""                  # B2B_RECORD_DIR (leer = aus)
    record_format: str = "pcm16"          # B2B_RECORD_FORMAT: pcm16 | float32
    # Zeitachse an Wanduhr
    drift_enable: bool = True             # B2B_DRIFT_ENABLE
    drift_base_s: float = 10.0            # B2B_DRIFT_BASE_S (Vorlauf, damit Lags positiv bleiben)
    drift_deadband_ms: float = 1.0        # B2B_DRIFT_DEADBAND_MS
    drift_slew_ms_per_s: float = 1.0      # B2B_DRIFT_SLEW_MS_PER_S
    # Diagnose
    debug_pos: bool = False               # B2B_DEBUG_POS: je Sekunde eine Zeile 'b2b-debug {...}' auf stderr

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LiveConfig":
        env = os.environ if env is None else env
        c = cls()
        for f in fields(cls):
            key = "B2B_" + f.name.upper()
            if key not in env or env[key] == "":
                continue
            raw = env[key]
            cur = getattr(c, f.name)
            if f.name == "roles":
                val = {}
                for part in raw.split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        val[k.strip()] = v.strip()
            elif f.name == "ignore_clients":
                val = tuple(x.strip() for x in raw.split(",") if x.strip())
            elif isinstance(cur, bool):
                val = _b(raw)
            elif isinstance(cur, int):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            else:
                val = raw
            setattr(c, f.name, val)
        if c.default_role not in ("dj", "listener"):
            raise ValueError("B2B_DEFAULT_ROLE muss dj oder listener sein")
        for r in c.roles.values():
            if r not in ("dj", "listener"):
                raise ValueError("B2B_ROLES: Rolle dj oder listener")
        if c.chunk < 256 or c.chunk % 64:
            raise ValueError("B2B_CHUNK: >= 256 und Vielfaches von 64")
        return c
