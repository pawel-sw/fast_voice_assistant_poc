import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from jarvis.remote_asr import RemoteASR
from jarvis.scheduling import FairGate


def test_asr_interleaved_states_do_not_mix():
    class Model:
        def init_streaming_state(self, **kwargs):
            return SimpleNamespace(text='', buffer=np.zeros(0))

        def streaming_transcribe(self, samples, state, **kwargs):
            state.text += ''.join(str(int(n)) for n in samples)

        def finish_streaming_transcribe(self, state, **kwargs):
            state.text += '!'

    asr = RemoteASR.__new__(RemoteASR)
    asr.model, asr.chunk_seconds = Model(), .48
    first, second = asr.new_state(), asr.new_state()
    asr.feed(np.array([1]), first)
    asr.feed(np.array([9]), second)
    assert asr.feed(np.array([2]), first, final=True) == '12!'
    assert asr.feed(np.array([8]), second, final=True) == '98!'


def test_fifo_gate_and_timeout_cleanup():
    gate = FairGate('test', timeout=2)
    order = []
    workers = []
    with gate:
        for i in range(3):
            def work(n=i):
                with gate:
                    order.append(n)
            thread = threading.Thread(target=work)
            thread.start()
            workers.append(thread)
            deadline = time.monotonic() + 1
            while len(gate.queue) != i + 1:
                assert time.monotonic() < deadline
                time.sleep(.001)
    for thread in workers:
        thread.join(3)
        assert not thread.is_alive()
    assert order == [0, 1, 2]
    gate.timeout = .01
    with gate:
        with pytest.raises(TimeoutError):
            with gate:
                pytest.fail('Must not enter occupied gate')
    assert not gate.queue
    with gate:
        pass


def test_gate_released_after_failure():
    gate = FairGate('test')
    with pytest.raises(ValueError):
        with gate:
            raise ValueError('model failed')
    with gate:
        assert gate.busy


def test_asr_uses_small_stream_and_final_token_budgets():
    from unittest.mock import Mock
    asr = RemoteASR.__new__(RemoteASR)
    asr.model = Mock()
    state = SimpleNamespace(text='fan on', buffer=np.zeros(0))
    assert asr.feed(np.zeros(2560), state, final=True) == 'fan on'
    assert asr.model.streaming_transcribe.call_args.kwargs['max_new_tokens'] == 4
    assert asr.model.finish_streaming_transcribe.call_args.kwargs['max_new_tokens'] == 8
    assert state.buffer.size == 160
