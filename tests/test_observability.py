import logging

import pytest

from jarvis.observability import PlainTranscript, Transcript, timed


def test_partial_transcript_never_enters_diagnostic_log(caplog):
    caplog.set_level(logging.INFO)
    transcript = Transcript('test')
    transcript.update('turn')
    transcript.update('turn on the')
    transcript.update('turn on', provisional=True)
    transcript.update('turn on the')
    transcript.update('turn off the lights', final=True)
    assert not caplog.records
    assert transcript.words == ['turn', 'off', 'the', 'lights']


def test_failed_calls_are_timed_and_still_raise(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(TimeoutError):
        with timed('test.request'):
            raise TimeoutError('not logged as credentials or request body')
    assert any('test.request' in r.message and 'outcome=TimeoutError' in r.message for r in caplog.records)


def test_high_frequency_timings_use_dedicated_logger(caplog):
    caplog.set_level(logging.INFO)
    with timed('openwakeword.predict', detail=True):
        pass
    assert len(caplog.records) == 1
    assert caplog.records[0].name == 'jarvis.timing'


def test_plain_wake_lines_corrections_continuations_and_json(tmp_path):
    path = tmp_path / 'transcript.txt'
    transcript = PlainTranscript(path)
    transcript.update('ignored', 'no wake')
    assert path.read_bytes() == b''
    transcript.wake()
    transcript.update('one', 'Hey Jarvis turn on')
    assert path.read_text() == 'Hey Jarvis turn on'
    transcript.update('one', 'Hey Jarvis turn off')
    transcript.update('two', 'the kitchen lights')
    transcript.command('Kitchen_Lights', 'OFF')
    line = 'Hey Jarvis turn off the kitchen lights [{"item":"Kitchen_Lights","command":"OFF"}]'
    assert path.read_text() == line
    transcript.wake()
    transcript.update('three', 'Hey Jarvis café')
    transcript.update('three', 'Hey Jarvis')
    assert path.read_text(encoding='utf-8') == line + '\nHey Jarvis'


def test_plain_restart_preserves_history_and_marks_failed_delivery(tmp_path):
    path = tmp_path / 'transcript.txt'
    path.write_text('old wake session', encoding='utf-8')
    transcript = PlainTranscript(path)
    transcript.wake()
    transcript.update('one', 'turn on lights')
    transcript.command('Kitchen_Lights', 'ON', error='Timeout')
    assert path.read_text() == 'old wake session\nturn on lights [{"item":"Kitchen_Lights","command":"ON","error":"Timeout"}]'


def test_unverified_candidates_do_not_pollute_clean_log(tmp_path, monkeypatch):
    from jarvis import observability as obs
    sink = PlainTranscript(tmp_path / 'transcript.txt')
    monkeypatch.setattr(obs, 'plain_transcript', sink)
    monkeypatch.setattr(obs, 'candidate_segments', None)
    obs.plain_candidate()
    Transcript('1').update('unrelated speech')
    obs.plain_cancel_candidate()
    assert sink.path.read_bytes() == b''
    obs.plain_candidate()
    Transcript('2').update('Hey Jarvis turn')
    obs.plain_confirm_candidate()
    assert sink.path.read_text() == 'Hey Jarvis turn'
