import numpy as np
import pytest

from b2b_relay.sim.synth import Track

FS = 48000


@pytest.fixture(scope="session")
def music() -> np.ndarray:
    """~83 s synthetischer DnB-Track (Mono, float32)."""
    return Track(seed=5, bars=60).mix


def run_mixer(mixer, feeds: dict, total: int, block: int = 512, start: int = 0, hook=None):
    """Feeds blockweise durch den Mixer schicken; liefert die Ausgabe (Länge total-start)."""
    outs = []
    for k in range(start, total - block + 1, block):
        outs.append(mixer.process({n: f[k:k + block] for n, f in feeds.items()}))
        if hook is not None:
            hook(k, mixer)
    return np.concatenate(outs)
