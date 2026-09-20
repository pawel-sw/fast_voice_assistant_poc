from unittest.mock import Mock
from concurrent.futures import Future
from jarvis.control import Catalog, Action
from jarvis.remote_rpc import RemotePlanner
from jarvis.groq_fallback import GroqFallback
from test_client_brain import make_brain


def planner():
    catalog = Catalog([{'name': 'Kitchen_Light', 'label': 'Kitchen light', 'type': 'Dimmer'}], {})
    result = RemotePlanner(catalog, {'groq_token': 'test-secret', 'min_confidence': .8})
    result.groq = Mock(token='test-secret')
    result.groq.repair.return_value = 'turn off kitchen light'
    result.groq.propose.return_value = {'confidence': .95, 'function_calls': [
        {'name': 'switch_kitchen_light', 'arguments': {'value': 'OFF'}}]}
    return result


def test_success_skips_cloud():
    p = planner()
    p._needle = Mock(return_value=[Action('Kitchen_Light', 'OFF')])
    assert p.plan('turn off kitchen light', 'session')
    p.groq.repair.assert_not_called()
    p.groq.propose.assert_not_called()


def test_timeout_repairs_and_retries_once():
    p = planner()
    p._needle = Mock(side_effect=[TimeoutError(), [Action('Kitchen_Light', 'OFF')]])
    assert p.plan('turn of kitchen light', 'session') == [Action('Kitchen_Light', 'OFF')]
    assert p._needle.call_args_list[1].args == ('turn off kitchen light', 'session')
    p.groq.propose.assert_not_called()


def test_second_failure_uses_validated_groq_json():
    p = planner()
    p._needle = Mock(return_value=[])
    assert p.plan('turn of kitchen light', 'session') == [Action('Kitchen_Light', 'OFF')]
    assert p._needle.call_count == 2
    p.groq.propose.assert_called_once_with('turn of kitchen light', 'turn off kitchen light')


def test_cloud_cannot_bypass_validation():
    for response in [None, {'confidence': .1, 'function_calls': []},
        {'confidence': 1, 'function_calls': [{'name': 'invented_tool', 'arguments': {}}]},
        {'confidence': 1, 'function_calls': [{'name': 'switch_kitchen_light', 'arguments': {'value': 'ON'}}]},
        {'confidence': 1, 'function_calls': [{'name': 'switch_kitchen_light', 'arguments': {'value': 'OFF', 'extra': 1}}]}]:
        p = planner()
        p._needle = Mock(return_value=[])
        p.groq.propose.return_value = response
        assert p.plan('turn off kitchen light', 'session') == []


def test_missing_token_and_provider_errors():
    p = planner()
    p._needle = Mock(return_value=[])
    p.groq.token = ''
    assert p.plan('test', 'session') == []
    p.groq.repair.assert_not_called()
    p.groq.token = 'test'
    p.groq.repair.side_effect = TimeoutError()
    p.groq.propose.side_effect = ValueError('bad JSON')
    assert p.plan('turn off kitchen light', 'session') == []


def test_late_fallback_is_discarded_after_continuation():
    brain, api, speaker = make_brain()
    brain.start()
    old = Future()
    brain.current.plan = old
    brain.current.planned_revision = 0
    brain.current.revision = 1
    brain.current.pending = 'updated request'
    old.set_result([Action('Kitchen_Light', 'OFF')])
    brain.tick()
    assert not api.calls


def test_request_uses_config_token_prompt_and_model(monkeypatch):
    p = planner()
    fallback = GroqFallback({'groq_token': 'private-test-key'}, p.catalog)
    response = Mock(ok=True)
    response.json.return_value = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'turn off kitchen light'}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr('jarvis.groq_fallback.requests.post', post)
    assert fallback.repair('turn of kitchen light') == 'turn off kitchen light'
    kwargs = post.call_args.kwargs
    assert kwargs['headers']['Authorization'] == 'Bearer private-test-key'
    assert kwargs['json']['model'] == 'openai/gpt-oss-120b'
    assert 'PHONETIC SIMILARITY' in kwargs['json']['messages'][0]['content']
    assert 'private-test-key' not in str(kwargs['json'])


def test_watch_log_shows_repair_and_compact_calls(tmp_path, monkeypatch):
    from jarvis import observability as obs
    path = tmp_path / 'live.txt'
    monkeypatch.setattr(obs, 'plain_transcript', obs.PlainTranscript(path))
    brain, api, speaker = make_brain()
    p = planner()
    p._needle = Mock(return_value=[])
    brain.planner = p
    session = brain.start()
    brain.transcript({'session': session, 'utterance': 0, 'text': 'turn of kitchen light', 'final': True})
    brain.tick()
    line = path.read_text(encoding='utf-8')
    assert '[groq repair: turn off kitchen light]' in line
    assert '[groq call: [{"name":"switch_kitchen_light","arguments":{"value":"OFF"}}]]' in line
    assert 'pending' not in line and '\n' not in line
    assert api.calls == [('Kitchen_Light', 'OFF')]
    brain.start()
    obs.Transcript(0).update('next request')
    assert path.read_text(encoding='utf-8').splitlines()[-1] == 'next request'


def test_watch_discards_stale_feedback(tmp_path, monkeypatch):
    from jarvis import observability as obs
    path = tmp_path / 'live.txt'
    monkeypatch.setattr(obs, 'plain_transcript', obs.PlainTranscript(path))
    brain, _, _ = make_brain()
    brain.start()
    old = brain.current
    brain.start()
    old.feedback.put((0, 'groq repair', 'old session'))
    brain.current.revision = 2
    brain.current.feedback.put((1, 'groq repair', 'old revision'))
    brain.current.feedback.put((2, 'groq repair', 'pending'))
    brain.tick()
    assert path.read_text(encoding='utf-8') == '[groq repair: pending]'
