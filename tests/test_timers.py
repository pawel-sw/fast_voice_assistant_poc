from pathlib import Path
from unittest.mock import Mock
import queue
import threading
import pytest
from jarvis.timers import TimerAction, TimerService, parse_timer
from jarvis.remote_rpc import RemotePlanner, SpeechOutput
from test_client_brain import make_brain, CATALOG


@pytest.mark.parametrize('text,seconds', [
    ('set timer to 10 seconds.', 10), ('set a timer to ten seconds', 10),
    ('set a timer for five minutes', 300), ('start a 90 second timer', 90),
    ('timer for one hour and twenty five minutes', 5100),
    ('set a timer for 1 minute, 30 seconds', 90),
    ('set a five-minute timer', 300), ('timer for half an hour', 1800),
    ('timer for one and a half minutes', 90), ('timer for 1.5 minutes', 90),
    ('set timer for a minute', 60), ('timer for twenty four hours', 86400)])
def test_duration_commands(text, seconds):
    assert parse_timer(text) == TimerAction('start', seconds)


@pytest.mark.parametrize('text', ['timer for zero seconds', 'timer for 25 hours',
    'set a timer', 'timer for five', 'timer for two three minutes', 'timer for -3 minutes',
    'timer for 1 minute and turn on kitchen light'])
def test_invalid_durations_do_not_start(text):
    assert parse_timer(text) == TimerAction('invalid')


def test_other_commands_not_intercepted():
    for text in ('turn off kitchen light', 'stop living room blinds', "don't set a timer for five minutes"):
        assert parse_timer(text) is None
    assert parse_timer('cancel timer') == TimerAction('cancel')
    assert parse_timer('stop') == TimerAction('stop')


def service(tmp_path):
    now = [0]
    timer = TimerService(Mock(), Mock(), path=tmp_path/'timer.json', clock=lambda: now[0],
                         wall=lambda: 1000+now[0], start_worker=False)
    return timer, now


def test_countdown_expiry_cancel_and_restore(tmp_path):
    timer, now = service(tmp_path)
    try:
        timer.apply(TimerAction('start', 65))
        assert timer.display() == '00:01:05'
        now[0] = 30
        assert timer.remaining() == 35
        restored = TimerService(Mock(), Mock(), path=timer.path, clock=lambda: now[0],
                                wall=lambda: 1030, start_worker=False)
        assert restored.remaining() == 35
        restored.close()
        now[0] = 65
        timer.tick()
        assert timer.is_ringing and timer.display() == '00:00:00 - Ringing'
        assert timer.events.get_nowait() == 'expired; ringing'
        timer.apply(TimerAction('stop'))
        assert not timer.is_ringing and timer.display() == '00:00:00'
        assert timer.ring_cancel.is_set()
    finally:
        timer.close()


def test_cancel_before_expiry_and_replacement(tmp_path):
    timer, now = service(tmp_path)
    timer.apply(TimerAction('start', 60))
    assert 'replaced' in timer.apply(TimerAction('start', 10))
    assert timer.remaining() == 10
    timer.apply(TimerAction('cancel'))
    now[0] = 80
    timer.tick()
    assert not timer.is_ringing and timer.deadline is None
    assert timer.events.empty()
    timer.close()


def test_stop_does_not_cancel_countdown_and_dry_run_does_not_save(tmp_path):
    timer, _ = service(tmp_path)
    timer.apply(TimerAction('start', 10), dry_run=True)
    assert not timer.path.exists()
    timer.apply(TimerAction('start', 10))
    assert timer.apply(TimerAction('stop')) == 'No timer is ringing.'
    assert timer.remaining() == 10
    timer.close()


def test_timer_planning_never_calls_models():
    planner = RemotePlanner(CATALOG, {'min_confidence': .8})
    planner._needle = Mock(side_effect=AssertionError('Must not call Needle'))
    assert planner.plan('set a timer for 5 minutes', 'x') == [TimerAction('start', 300)]


def test_timer_to_duration_after_groq_repair():
    planner = RemotePlanner(CATALOG, {'min_confidence': .8, 'groq_token': 'test'})
    planner._needle = Mock(return_value=[])
    planner.groq = Mock(token='test')
    planner.groq.repair.return_value = 'set timer to 10 seconds'
    assert planner.plan('Alex set timer to 10 seconds.', 'x') == [TimerAction('start', 10)]
    planner.groq.propose.assert_not_called()


def test_wakeless_timer_session_can_only_cancel_or_stop(tmp_path):
    brain, api, speaker = make_brain()
    brain.timer, _ = service(tmp_path)
    brain.timer.ringing = True
    session = brain.start(timer_only=True)
    brain.transcript({'session': session, 'utterance': 0, 'text': 'turn on kitchen light', 'final': True})
    brain.tick()
    assert not api.calls
    session = brain.start(timer_only=True)
    brain.transcript({'session': session, 'utterance': 0, 'text': 'Stop!', 'final': True})
    brain.tick()
    assert not brain.timer.is_ringing
    assert speaker.replies == ['Timer stopped.']
    brain.timer.close()


def test_chime_is_bounded_pcm_and_cancelled_playback_is_silent():
    chunks = []
    speaker = SpeechOutput({}, sink=chunks.append)
    speaker.chime(threading.Event())
    assert sum(map(len, chunks)) == 24000*2*1.8
    assert max(map(len, chunks)) <= 2400
    cancelled = threading.Event()
    cancelled.set()
    chunks.clear()
    speaker.chime(cancelled)
    assert not chunks
