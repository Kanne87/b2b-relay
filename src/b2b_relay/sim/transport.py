"""Transportmodell: konstante Latenz, Jitter, Paketverlust und Jitter-Puffer.

Jedes Paket (``packet`` Samples) bekommt eine zufällige Zusatzverzögerung (exponentialverteilt).
Der Jitter-Puffer hat die Länge ``jitter_s``: Pakete, die später als Puffer ankommen, gelten als
Verlust, ebenso zufällig verlorene Pakete. Verlust = Nullen. Die Latenz je Richtung ist durch den
Puffer konstant (``one_way_s`` enthält den Puffer), deshalb genügt es, die Verlustmaske zu
erzeugen; die Laufzeit selbst steckt in der Lag-Funktion des DJ-Modells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PACKET = 128    # Samples je Paket (2,67 ms bei 48 kHz), wie ein JACK-Puffer


@dataclass
class Link:
    one_way_s: float      # feste Latenz einschließlich Jitter-Puffer
    jitter_s: float
    loss: float
    packet: int = PACKET

    def __post_init__(self) -> None:
        if self.jitter_s > self.one_way_s + 1e-12:
            raise ValueError("Der Jitter-Puffer darf nicht größer als die Einweg-Latenz sein "
                             f"({self.jitter_s * 1e3:.1f} ms > {self.one_way_s * 1e3:.1f} ms)")

    def loss_mask(self, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        """True = Paket verloren (zufällig oder zu spät für den Jitter-Puffer)."""
        n = n_samples // self.packet + 2
        dropped = rng.random(n) < self.loss
        if self.jitter_s > 0:
            late = rng.exponential(self.jitter_s / 8.0, n) > self.jitter_s
        else:
            late = np.zeros(n, dtype=bool)
        return dropped | late, {"packets": int(n), "dropped": int(dropped.sum()),
                                "late": int(late.sum())}
