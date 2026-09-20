from datetime import datetime
from unittest.mock import Mock
import pytest

from jarvis.clock_queries import ClockQuery, clock_answer, parse_clock
from jarvis.remote_rpc import RemotePlanner
from test_client_brain import CATALOG, make_brain


@pytest.mark.parametrize('text,kind', [
    ('what time it is?', 'time'), ('What time is it?', 'time'),
    ("what's the time", 'time'), ('please tell me what time it is', 'time'),
    ('what day is it', 'date'), ('what day is today?', 'date'),
    ("what's today's date?", 'date'), ('what is the date today please', 'date'),
])
def test_clock_intents(text, kind):
    planner = RemotePlanner(CATALOG, {'min_confidence': .8})
    planner._try_needle = Mock(side_effect=AssertionError('Clock must bypass inference'))
    assert planner.plan(text, 'session') == [ClockQuery(kind)]


@pytest.mark.parametrize('text', ['how much time is left on the timer', 'set timer to ten seconds',
                               'turn on daytime light', 'what time is it in Tokyo',
                               'what day is it and turn on the light'])
def test_unrelated_or_compound_requests_are_not_clock(text):
    assert parse_clock(text) is None


def test_date_and_midnight_noon_format():
    assert clock_answer(ClockQuery('date'), datetime(2026, 9, 20)) == 'Sunday September 20.'
    assert clock_answer(ClockQuery('time'), datetime(2026, 9, 20, 0, 5)) == 'It is 12:05 AM.'
    assert clock_answer(ClockQuery('time'), datetime(2026, 9, 20, 12, 0)) == 'It is 12:00 PM.'
    assert clock_answer(ClockQuery('time'), datetime(2026, 9, 20, 15, 9)) == 'It is 3:09 PM.'


def test_date_is_logged_and_spoken_without_home_commands(tmp_path, monkeypatch):
    from jarvis import observability as obs
    path = tmp_path / 'live.txt'
    monkeypatch.setattr(obs, 'plain_transcript', obs.PlainTranscript(path))
    monkeypatch.setattr('jarvis.client_brain.clock_answer', lambda query: 'Sunday September 20.')
    brain, api, speaker = make_brain(action=ClockQuery('date'))
    session = brain.start()
    brain.transcript({'session': session, 'utterance': 0, 'text': 'Alexa what day is it', 'final': True})
    brain.tick()
    assert speaker.replies == ['Sunday September 20.']
    assert '[date: Sunday September 20.]' in path.read_text()
    assert not api.calls
    brain.close()
