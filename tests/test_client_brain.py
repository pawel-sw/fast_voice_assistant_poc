import threading
import time
from concurrent.futures import Future

from jarvis.client_brain import ClientBrain
from jarvis.control import Action, Catalog, TemperatureQuery
from jarvis.remote_rpc import validate_response
from jarvis.speech import confirmation

CONFIG = {'continuation_seconds': 5, 'min_confidence': .8}
CATALOG = Catalog([
    {'name': 'Kitchen_Light', 'type': 'Dimmer', 'label': 'Kitchen light'},
    {'name': 'Outside_Temperature', 'type': 'Number:Temperature'}], {})


class ImmediatePool:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future
    def shutdown(self, **kwargs):
        pass


class API:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail
    def command(self, item, value):
        self.calls.append((item, value))
        if self.fail:
            raise TimeoutError()
    def temperature(self, item):
        return '12 °C'


class Speaker:
    def __init__(self):
        self.replies = []
    def speak(self, text, session, cancelled):
        self.replies.append(text)


class Planner:
    def __init__(self, action):
        self.action = action
    def plan(self, text, session):
        return [self.action]


def make_brain(action=Action('Kitchen_Light', 'ON'), fail=False):
    api, speaker = API(fail), Speaker()
    brain = ClientBrain(api, CATALOG, Planner(action), speaker, CONFIG)
    brain.pool.shutdown()
    brain.pool = ImmediatePool()
    return brain, api, speaker


def test_replies_use_real_outcomes():
    assert confirmation(Action('Kitchen_Light', 'ON'), CATALOG) == 'Kitchen light turned on.'
    assert confirmation(Action('Kitchen_Light', '35'), CATALOG) == 'Kitchen light set to 35 percent.'
    assert confirmation(TemperatureQuery('Outside_Temperature', 'outside'), CATALOG, '12 °C') == 'The temperature outside is 12 degrees Celsius.'


def test_client_executes_once_and_speaks_after_success():
    brain, api, speaker = make_brain()
    session = brain.start()
    event = {'session': session, 'utterance': 0, 'text': 'Hey Jarvis turn on kitchen light', 'final': True}
    brain.transcript(event)
    brain.tick()
    brain.transcript(event)
    brain.tick()
    assert api.calls == [('Kitchen_Light', 'ON')]
    assert speaker.replies == ['Kitchen light turned on.']


def test_http_failure_never_speaks_success_or_retries():
    brain, api, speaker = make_brain(fail=True)
    session = brain.start()
    brain.transcript({'session': session, 'utterance': 0, 'text': 'turn on kitchen light', 'final': True})
    brain.tick()
    brain.tick()
    assert len(api.calls) == 1
    assert 'could not confirm' in speaker.replies[0]
    assert 'turned on' not in speaker.replies[0]


def test_unverified_wake_and_stale_results_do_not_act():
    brain, api, speaker = make_brain()
    old = brain.start()
    current = brain.start(tentative=True)
    brain.transcript({'session': old, 'utterance': 0, 'text': 'Hey Jarvis turn on kitchen light', 'final': True})
    brain.transcript({'session': current, 'utterance': 0, 'text': 'turn on kitchen light', 'final': True})
    brain.tick()
    assert not api.calls and not speaker.replies


def test_client_rejects_ungrounded_needle_switch():
    schemas = CATALOG.candidates('turn off kitchen light')
    response = {'confidence': 1, 'function_calls': [{'name': schemas[0]['name'], 'arguments': {'value': 'ON'}}]}
    assert validate_response(CATALOG, 'turn off kitchen light', schemas, response, .8) == []


def test_superseded_inflight_plan_cannot_execute():
    brain, api, speaker = make_brain()
    brain.start()
    future = Future()
    brain.current.plan = future
    brain.start()
    future.set_result([Action('Kitchen_Light', 'ON')])
    brain.tick()
    assert not api.calls


def test_alexa_confirms_candidate_and_is_removed_from_request():
    from unittest.mock import Mock
    brain, api, _ = make_brain()
    brain.planner.plan = Mock(return_value=[Action('Kitchen_Light', 'ON')])
    session = brain.start(tentative=True)
    brain.transcript({'session': session, 'utterance': 0,
                      'text': 'Alexa, turn on kitchen light', 'final': True})
    brain.tick()
    brain.planner.plan.assert_called_once_with('turn on kitchen light', session)
    assert api.calls == [('Kitchen_Light', 'ON')]


def test_alexa_requires_word_boundary():
    brain, api, _ = make_brain()
    session = brain.start(tentative=True)
    brain.transcript({'session': session, 'utterance': 0,
                      'text': 'Alexander turn on kitchen light', 'final': True})
    brain.tick()
    assert not api.calls
