"""Planning is tested without importing HA's Python-3.14 runtime."""
import asyncio
import importlib
from pathlib import Path
import sys
import types

import pytest

PACKAGE = 'jarvis_ha_test'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'custom_components/jarvis_conversation')]
sys.modules[PACKAGE] = package
module = importlib.import_module(PACKAGE + '.planner')
SCHEMAS = [{'name': 'HassTurnOn', 'parameters': {'type': 'object'}}]
GOOD = {'confidence': 0.95, 'function_calls': [{'name': 'HassTurnOn', 'arguments': {'name': 'Desk'}}]}


def validate(name, args):
    if set(args) != {'name'} or not isinstance(args['name'], str):
        raise ValueError('Invalid arguments')
    return args


class FakePlanner(module.Planner):
    def __init__(self, needle, groq=(), enabled=True):
        super().__init__(None, {'groq_api_key': 'test' if enabled else ''})
        self.local, self.cloud, self.steps = iter(needle), iter(groq), []

    async def needle(self, text, schemas, prompt):
        self.steps.append(('needle', text))
        value = next(self.local)
        if isinstance(value, BaseException):
            raise value
        return value

    async def groq(self, instruction, payload):
        self.steps.append(('groq', payload['transcript']))
        value = next(self.cloud)
        if isinstance(value, BaseException):
            raise value
        return value

    def run(self):
        return asyncio.run(self.plan('turn on desk', SCHEMAS, 'Desk is a light', [], validate))


def test_local_path_never_calls_cloud():
    planner = FakePlanner([GOOD])
    assert planner.run().stage == 'needle'
    assert [s[0] for s in planner.steps] == ['needle']


def test_repair_then_retry():
    planner = FakePlanner([{}, GOOD], [{'corrected': 'turn on Desk'}])
    assert planner.run().stage == 'needle-retry'
    assert planner.steps[-1] == ('needle', 'turn on Desk')


def test_full_fallback():
    planner = FakePlanner([{}, {}], [{'corrected': 'turn on Desk'}, GOOD])
    assert planner.run().stage == 'groq'
    assert [s[0] for s in planner.steps] == ['needle', 'groq', 'needle', 'groq']


def test_local_outage_does_not_retry_unreachable_worker():
    planner = FakePlanner([TimeoutError()], [{'corrected': 'turn on Desk'}, GOOD])
    assert planner.run().stage == 'groq'
    assert [s[0] for s in planner.steps] == ['needle', 'groq', 'groq']


def test_optional_cloud_and_failure():
    assert not FakePlanner([{}], enabled=False).run().calls
    planner = FakePlanner([{}, {}], [ValueError(), ValueError()])
    assert not planner.run().calls


def test_cancellation_never_falls_back():
    planner = FakePlanner([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        planner.run()
    assert len(planner.steps) == 1


@pytest.mark.parametrize('response', [
    {'confidence': float('nan'), 'function_calls': GOOD['function_calls']},
    {'confidence': True, 'function_calls': GOOD['function_calls']},
    {'confidence': 0.2, 'function_calls': GOOD['function_calls']},
    {'confidence': 1, 'function_calls': [{'name': 'invented', 'arguments': {}}]},
    {'confidence': 1, 'function_calls': [{'name': 'HassTurnOn', 'arguments': {'extra': 1}}]},
    {'confidence': 1, 'function_calls': GOOD['function_calls'] * 2},
    {'confidence': 1, 'function_calls': GOOD['function_calls'] + [{'name': 'invented', 'arguments': {}}]},
])
def test_invalid_proposals_reject_entire_batch(response):
    assert module.validate_plan(response, SCHEMAS, validate, 'groq') is None


def test_clarification_has_no_actions():
    planner = FakePlanner([{}, {}], [{'corrected': 'turn on desk'},
        {'confidence': 1, 'function_calls': [], 'reply': 'Which desk light?'}])
    result = planner.run()
    assert not result.calls and result.reply == 'Which desk light?'


def test_tool_result_errors_are_not_success():
    assert module.result_speech({'error': 'timeout'})[1]
    assert module.result_speech({'response_type': 'error', 'speech': {'plain': {'speech': 'Done'}}})[1]
    assert module.result_speech({'data': {'failed': [{'name': 'Desk'}]}})[1]
    assert module.result_speech({'speech': {'plain': {'speech': 'Done'}}, 'data': {'failed': [{'name': 'Desk'}]}})[1]
    assert module.result_speech({'speech': {'plain': {'speech': 'It is 20 degrees.'}}}) == ('It is 20 degrees.', False)


def test_backend_url_is_bounded_to_rpc():
    assert module.rpc_url('ws://example.invalid:8765/asr?x=1') == 'ws://example.invalid:8765/rpc'
    with pytest.raises(ValueError):
        module.rpc_url('http://example.invalid')


def test_read_queries_and_conditions_do_not_offer_write_tools():
    schemas = [*SCHEMAS, {'name': 'homeassistant__GetLiveContext'}]
    for text in ['Is the desk on?', 'What is the temperature?', 'If it is cold turn on the desk', 'Explain rainbows']:
        assert [s['name'] for s in module.select_tools(text, schemas)] == ['homeassistant__GetLiveContext']
    assert 'HassTurnOn' in [s['name'] for s in module.select_tools('turn on Desk', schemas)]


def test_live_context_is_spoken_as_values_not_json():
    result = {'success': True, 'result': 'Live Context: Devices\n- names: Kitchen\n  state: "21"\n  attributes:\n    unit_of_measurement: C\n'}
    assert module.result_speech(result) == ('Kitchen: 21 C', False)
