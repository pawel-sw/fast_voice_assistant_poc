import json
from pathlib import Path
import numpy as np
from jarvis.remote_session import AudioSession
from jarvis.control import Action

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())


class ASR:
    text = 'Hey Jarvis turn off guest bedroom light.'
    def reset(self):
        pass
    def feed(self, samples, final=False):
        return self.text


class Planner:
    def plan(self, text):
        return [Action('guest', 'OFF')] if 'turn off' in text else []


def session(asr=None, tentative=False):
    events, actions = [], []
    obj = AudioSession(asr or ASR(), Planner(), actions.append, CONFIG, events.append, tentative)
    return obj, events, actions


def test_preroll_silence_does_not_finish_before_wake():
    obj, events, actions = session(tentative=True)
    obj.feed(bytes(640*150), preroll=True)
    assert not obj.closed and not actions
    obj.feed(bytes(640*40))
    assert actions == [Action('guest', 'OFF')]
    assert [e['type'] for e in events].count('confirmed') == 1
    assert events[-1]['type'] == 'done'


def test_unconfirmed_candidate_cannot_execute():
    asr = ASR()
    asr.text = 'turn off guest bedroom light.'
    obj, events, actions = session(asr, tentative=True)
    obj.feed(bytes(640*150), preroll=True)
    obj.feed(bytes(640*40))
    assert obj.closed and not actions


def test_closed_session_ignores_queued_audio():
    obj, events, actions = session()
    obj.feed(bytes(640*150), preroll=True)
    obj.feed(bytes(640*40))
    obj.feed(bytes(640*500))
    assert len(actions) == 1


def test_partial_request_waits_for_continuation_then_expires():
    asr = ASR()
    asr.text = 'Hey Jarvis'
    obj, events, actions = session(asr)
    obj.feed(bytes(640*150), preroll=True)
    obj.feed(bytes(640*40))
    assert not obj.closed
    obj.feed(bytes(640*300))
    assert obj.closed and not actions


def test_bad_pcm_rejected():
    import pytest
    obj, _, _ = session()
    with pytest.raises(ValueError):
        obj.feed(b'x')
