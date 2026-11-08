"""Minimaler JACK-Ersatz für Tests ohne jackd (nur die von ``live/jackio.py`` genutzte Teilmenge)."""

from __future__ import annotations

import numpy as np


class FakePort:
    def __init__(self, name: str, is_output: bool, frames: int) -> None:
        self.name = name
        self.is_output = is_output
        self.is_input = not is_output
        self.buf = np.zeros(frames, dtype=np.float32)

    def get_array(self) -> np.ndarray:
        return self.buf


class _Reg:
    def __init__(self, client: "FakeClient", is_output: bool) -> None:
        self.c, self.out = client, is_output

    def register(self, short: str) -> FakePort:
        p = FakePort(f"{self.c.name}:{short}", self.out, self.c.blocksize)
        self.c.world.ports[p.name] = p
        return p


class FakeJack:
    """Die 'JACK-Welt': Ports und Verbindungen aller Clients."""

    def __init__(self, frames: int = 256, fs: int = 48000) -> None:
        self.ports: dict[str, FakePort] = {}
        self.conns: set[tuple[str, str]] = set()
        self.frames, self.fs = frames, fs

    def add_remote(self, client: str, n: int = 2) -> None:
        for k in range(1, n + 1):
            self.ports[f"{client}:receive_{k}"] = FakePort(f"{client}:receive_{k}", True, self.frames)
            self.ports[f"{client}:send_{k}"] = FakePort(f"{client}:send_{k}", False, self.frames)

    def remove_remote(self, client: str) -> None:
        for n in [n for n in self.ports if n.startswith(client + ":")]:
            del self.ports[n]
        self.conns = {c for c in self.conns if not c[0].startswith(client + ":") and not c[1].startswith(client + ":")}

    def factory(self, name: str) -> "FakeClient":
        return FakeClient(self, name)

    def sources_of(self, dst: str) -> set[str]:
        return {a for a, b in self.conns if b == dst}


class FakeClient:
    def __init__(self, world: FakeJack, name: str) -> None:
        self.world, self.name = world, name
        self.samplerate, self.blocksize = world.fs, world.frames
        self.inports, self.outports = _Reg(self, False), _Reg(self, True)
        self.cb = self.xcb = self.scb = None
        self.active = False

    def set_process_callback(self, f): self.cb = f
    def set_xrun_callback(self, f): self.xcb = f
    def set_shutdown_callback(self, f): self.scb = f
    def activate(self): self.active = True
    def deactivate(self): self.active = False
    def close(self): pass
    def cpu_load(self): return 1.5

    def get_ports(self, name_pattern="", is_audio=False, is_input=False, is_output=False):
        import re
        out = []
        for n, p in self.world.ports.items():
            if name_pattern and not re.search(name_pattern, n):
                continue
            if is_input and not p.is_input:
                continue
            if is_output and not p.is_output:
                continue
            out.append(p)
        return out

    def get_all_connections(self, port):
        name = port if isinstance(port, str) else port.name
        return [self.world.ports[a] for a, b in self.world.conns if b == name and a in self.world.ports]

    def connect(self, a, b): self.world.conns.add((a, b))
    def disconnect(self, a, b): self.world.conns.discard((a, b))
