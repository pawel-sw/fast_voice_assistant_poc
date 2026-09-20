import json
import threading
from unittest.mock import Mock
import pytest
from clients.inference_client import speech_messages


def test_receives_end_before_slow_consumer_finishes_audio():
    end_received = threading.Event()
    packets = iter([b'audio'] * 80 + [json.dumps({'type': 'tts_end'})])
    def recv(**kwargs):
        message = next(packets)
        if isinstance(message, str):
            end_received.set()
        return message
    ws = Mock(recv=recv)
    stream = speech_messages(ws)
    assert next(stream) == b'audio'
    # Consumer is paused, but the network must continue to drain to tts_end.
    assert end_received.wait(1)
    remaining = list(stream)
    assert remaining[:-1] == [b'audio'] * 79
    assert json.loads(remaining[-1])['type'] == 'tts_end'
    ws.close.assert_called_once()


def test_response_memory_limit_and_disconnect_surface_errors():
    ws = Mock()
    ws.recv.return_value = b'12345'
    with pytest.raises(ValueError, match='buffer limit'):
        list(speech_messages(ws, max_bytes=4))
    ws.recv.side_effect = ConnectionError('unexpected EOF')
    with pytest.raises(ConnectionError):
        list(speech_messages(ws))


def test_cancellation_closes_socket_and_unblocks_receiver():
    cancelled, closed = threading.Event(), threading.Event()
    def recv(**kwargs):
        closed.wait(2)
        raise ConnectionError('closed')
    ws = Mock(recv=recv, close=closed.set)
    cancelled.set()
    assert list(speech_messages(ws, cancelled=cancelled)) == []
    assert closed.is_set()
