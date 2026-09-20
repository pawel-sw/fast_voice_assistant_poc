import logging
from datetime import datetime
import pytest
from jarvis.observability import timestamp, timestamp_span


def test_retrospective_audio_event_preserves_capture_time(caplog):
    with caplog.at_level(logging.INFO, logger='jarvis.timing'):
        timestamp('last_speech_audio', 'session-test', at=123.456, utterance=2)
    message = caplog.records[-1].getMessage()
    assert 'monotonic=123.456000 session=session-test utterance=2' in message
    stamp = message.split('timestamp=')[1].split()[0]
    assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 0


def test_failed_call_has_matching_done_event(caplog):
    with caplog.at_level(logging.INFO, logger='jarvis.timing'):
        with pytest.raises(TimeoutError):
            with timestamp_span('openhab', 'session-test'):
                raise TimeoutError()
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert 'EVENT openhab_start' in messages[0]
    assert 'EVENT openhab_done' in messages[1] and 'outcome=TimeoutError' in messages[1]
