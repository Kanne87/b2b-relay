"""Minimaler WAV-Schreiber (PCM16 oder IEEE-Float32), Kopf wird laufend nachgezogen.

Die Standardbibliothek ``wave`` kann kein Float; für den Mitschnitt der Programmausgabe (und die
Auswertung im Integrationstest) reicht dieser kleine Schreiber. Der Kopf wird alle ``flush_s``
Sekunden und beim Schließen mit den echten Längen überschrieben, damit auch eine nicht sauber
beendete Datei abspielbar bleibt.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


class WavWriter:
    def __init__(self, path: str | Path, sample_rate: int, channels: int = 2,
                 fmt: str = "pcm16", flush_s: float = 5.0) -> None:
        if fmt not in ("pcm16", "float32"):
            raise ValueError("fmt: pcm16 oder float32")
        self.path = Path(path)
        self.fs, self.ch, self.fmt = sample_rate, channels, fmt
        self.frames = 0
        self._flush_every = int(flush_s * sample_rate)
        self._since_flush = 0
        self._f = open(self.path, "wb")
        self._f.write(self._header(0))

    @property
    def _bytes_per_sample(self) -> int:
        return 2 if self.fmt == "pcm16" else 4

    def _header(self, frames: int) -> bytes:
        bps = self._bytes_per_sample
        data = frames * self.ch * bps
        tag = 1 if self.fmt == "pcm16" else 3
        return (b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE" + b"fmt " +
                struct.pack("<IHHIIHH", 16, tag, self.ch, self.fs, self.fs * self.ch * bps,
                            self.ch * bps, 8 * bps) + b"data" + struct.pack("<I", data))

    def write(self, chans: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
        x = np.stack(chans, axis=1)                  # (n, ch) verschachtelt
        if self.fmt == "pcm16":
            x = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
        else:
            x = x.astype("<f4")
        self._f.write(x.tobytes())
        self.frames += x.shape[0]
        self._since_flush += x.shape[0]
        if self._since_flush >= self._flush_every:
            self._patch()

    def _patch(self) -> None:
        pos = self._f.tell()
        self._f.seek(0)
        self._f.write(self._header(self.frames))
        self._f.seek(pos)
        self._f.flush()
        self._since_flush = 0

    def close(self) -> None:
        if not self._f.closed:
            self._patch()
            self._f.close()


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Liest PCM16/PCM24/PCM32/Float32 → (Samplerate, float32-Array (n, ch))."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("keine WAV-Datei")
    pos, fs, ch, tag, bits = 12, 0, 0, 0, 0
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            tag, ch, fs, _, _, bits = struct.unpack("<HHIIHH", raw[body:body + 16])
        elif cid == b"data":
            end = len(raw) if size in (0, 0xFFFFFFFF) or body + size > len(raw) else body + size
            data = raw[body:end]
            if tag == 3 and bits == 32:
                a = np.frombuffer(data[:len(data) // 4 * 4], dtype="<f4").astype(np.float32)
            elif tag == 1 and bits == 16:
                a = np.frombuffer(data[:len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
            elif tag == 1 and bits == 32:
                a = np.frombuffer(data[:len(data) // 4 * 4], dtype="<i4").astype(np.float32) / 2147483648.0
            elif tag == 1 and bits == 24:
                b = np.frombuffer(data[:len(data) // 3 * 3], dtype=np.uint8).reshape(-1, 3)
                v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) |
                     (b[:, 2].astype(np.int32) << 16))
                v = np.where(v >= 1 << 23, v - (1 << 24), v)
                a = v.astype(np.float32) / 8388608.0
            else:
                raise ValueError(f"WAV-Format nicht unterstützt (tag={tag}, bits={bits})")
            n = len(a) // ch
            return fs, a[:n * ch].reshape(n, ch)
        pos = body + size + (size & 1)
    raise ValueError("kein data-Chunk")
