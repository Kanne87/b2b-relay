import pytest

from b2b_relay.live.config import LiveConfig


def test_defaults_and_env_override():
    c = LiveConfig.from_env({})
    assert (c.sample_rate, c.api_port, c.target_buffer_s, c.default_role) == (48000, 8090, 3.0, "dj")
    c = LiveConfig.from_env({"B2B_API_PORT": "9000", "B2B_TARGET_BUFFER_S": "4.5", "B2B_AUTO_ACTIVE": "false",
                             "B2B_ROLES": "tv=listener, a=dj", "B2B_RECORD_DIR": "/rec"})
    assert c.api_port == 9000 and c.target_buffer_s == 4.5 and c.auto_active is False
    assert c.roles == {"tv": "listener", "a": "dj"} and c.record_dir == "/rec"


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        LiveConfig.from_env({"B2B_DEFAULT_ROLE": "boss"})
    with pytest.raises(ValueError):
        LiveConfig.from_env({"B2B_CHUNK": "100"})
