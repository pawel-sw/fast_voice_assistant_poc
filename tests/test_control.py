import pytest

from jarvis.control import Action, Catalog, Conversation, OpenHAB


@pytest.fixture
def catalog():
    return Catalog([
        {'name': 'Kitchen_Lights', 'type': 'Dimmer', 'label': 'Kitchen Lights'},
        {'name': 'Basement_Kitchen_Light', 'type': 'Dimmer', 'label': 'Basement Kitchen Light'},
        {'name': 'Sensor', 'type': 'Switch', 'tags': ['Switch'], 'stateDescription': {'readOnly': True}},
        {'name': 'Garden_Valve2', 'type': 'Switch', 'label': 'Garden Valve 2', 'tags': ['Switch']},
        {'name': 'Bedroom_Shade', 'type': 'Rollershutter', 'label': 'Bedroom Shade'},
    ], {})


def test_catalog_excludes_sensors_and_no_guessed_room(catalog):
    assert 'Sensor' not in catalog.items
    assert catalog.candidates('turn on') == []
    assert {s['name'] for s in catalog.candidates('turn on kitchen lights')} == {'switch_kitchen_lights'}
    assert {s['name'] for s in catalog.candidates('dim basement kitchen lights to 35 percent')} == {'level_basement_kitchen_light'}
    assert {s['name'] for s in catalog.candidates('turn on garden valve 2')} == {'switch_garden_valve2'}


@pytest.mark.parametrize('call', [
    {'name': 'unknown', 'arguments': {'value': 'ON'}},
    {'name': 'level_kitchen_lights', 'arguments': {'value': 101}},
    {'name': 'level_kitchen_lights', 'arguments': {'value': True}},
    {'name': 'switch_kitchen_lights', 'arguments': {'value': 'TOGGLE'}},
    {'name': 'switch_kitchen_lights', 'arguments': {'value': 'ON', 'url': 'elsewhere'}},
])
def test_reject_invalid_calls(catalog, call):
    with pytest.raises(ValueError):
        catalog.validate([call])


def test_shade_translation(catalog):
    assert catalog.validate([{'name': 'move_bedroom_shade', 'arguments': {'value': 'open'}}]) == [Action('Bedroom_Shade', 'UP')]


class PlannerStub:
    def __init__(self):
        self.calls = []

    def plan(self, text):
        self.calls.append(text)
        return [Action('Kitchen_Lights', 'ON')] if 'kitchen lights' in text.lower() else []


def test_pause_continuation_and_exactly_once():
    planner, sent = PlannerStub(), []
    conversation = Conversation(planner, sent.append)
    conversation.utterance('turn on kitchen lights', 0)
    assert not sent
    conversation.activate(1)
    conversation.utterance('Hey, Jarvis! Turn on', 1)
    conversation.utterance('the kitchen lights', 3)
    conversation.utterance('the kitchen lights', 4)
    assert len(sent) == 1
    assert planner.calls == ['Turn on', 'Turn on the kitchen lights']


def test_expired_session_and_embedded_wake_word():
    planner, sent = PlannerStub(), []
    conversation = Conversation(planner, sent.append)
    conversation.activate(1)
    conversation.utterance('Hey Jarvis', 1)
    conversation.utterance('kitchen lights', 8)
    conversation.utterance('hey jarvison kitchen lights', 9)
    assert not sent


def test_timeout_never_replays_command():
    planner = PlannerStub()
    def timeout(action):
        raise TimeoutError()
    conversation = Conversation(planner, timeout)
    conversation.activate(1)
    with pytest.raises(TimeoutError):
        conversation.utterance('Hey Jarvis kitchen lights', 1)
    assert conversation.pending is None


def test_transcript_cannot_activate_wake_session():
    sent = []
    conversation = Conversation(PlannerStub(), sent.append)
    conversation.utterance('Hey Jarvis turn on kitchen lights', 1)
    assert not sent
    conversation.activate(2)
    conversation.utterance('turn on kitchen lights', 3)
    assert sent == [Action('Kitchen_Lights', 'ON')]


def test_wake_removal_preserves_command_punctuation():
    planner = PlannerStub()
    conversation = Conversation(planner, lambda action: None)
    conversation.activate(0)
    conversation.utterance('Hey Jarvis, turn off guest bedroom light.', 1)
    assert planner.calls == ['turn off guest bedroom light.']


def test_weak_acoustic_candidate_requires_transcribed_wake_phrase():
    sent = []
    conversation = Conversation(PlannerStub(), sent.append)
    conversation.activate(0, tentative=True)
    conversation.utterance('turn on kitchen lights', 1)
    assert sent == []
    assert conversation.pending is None
    conversation.activate(2, tentative=True)
    conversation.utterance('Hey Jarvis turn on kitchen lights', 3)
    assert sent == [Action('Kitchen_Lights', 'ON')]


def test_planner_normalizes_sentence_and_rejects_unspoken_switch_action(catalog):
    from jarvis.control import Planner
    planner = Planner.__new__(Planner)
    planner.catalog, planner.minimum, planner.model = catalog, 0.8, None
    seen = []
    class Model:
        def __init__(self, **kwargs):
            pass
        def complete(self, text, **kwargs):
            seen.append(text)
            return {'confidence': 0.99, 'function_calls': [
                {'name': 'switch_kitchen_lights', 'arguments': {'value': 'OFF'}}]}
        def close(self):
            pass
    planner.factory = Model
    assert planner.plan('turn off kitchen lights') == [Action('Kitchen_Lights', 'OFF')]
    assert seen[-1] == 'turn off kitchen lights.'
    assert planner.plan('Tune of kitchen lights') == []


def test_command_endpoint(monkeypatch):
    api = OpenHAB('http://example.invalid')
    class Response:
        status_code = 202
        def raise_for_status(self):
            pass
    calls = []
    monkeypatch.setattr(api.session, 'post', lambda *a, **kw: calls.append((a, kw)) or Response())
    assert api.command('Kitchen_Lights', 'ON') == 202
    assert calls[0][0][0] == 'http://example.invalid/rest/items/Kitchen_Lights'
    assert calls[0][1]['data'] == b'ON'
