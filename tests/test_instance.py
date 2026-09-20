import pytest

from jarvis.instance import listener_lock


def test_only_one_listener_and_lock_released(tmp_path):
    path = tmp_path / 'listener.lock'
    with listener_lock(path):
        with pytest.raises(RuntimeError, match='already listening'):
            with listener_lock(path):
                pytest.fail('duplicate listener')
    with listener_lock(path):
        pass
