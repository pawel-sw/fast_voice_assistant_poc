import json

import pytest

from jarvis.control import Catalog, OpenHAB, TemperatureQuery, execute_action
from jarvis import observability


def test_sensor_query_catalog_is_explicit_and_read_only():
    catalog = Catalog([
        {'name': 'Outside_Temperature', 'type': 'Number:Temperature', 'stateDescription': {'readOnly': True}},
        {'name': 'Inside_Temperature', 'type': 'Number:Temperature', 'stateDescription': {'readOnly': True}},
    ], {})
    assert not catalog.items
    assert [s['name'] for s in catalog.candidates('temperature outside')] == ['get_outside_temperature']
    assert [s['name'] for s in catalog.candidates('how warm is it indoors')] == ['get_inside_temperature']
    assert not catalog.candidates('what is the temperature')
    assert catalog.validate([{'name': 'get_outside_temperature', 'arguments': {}}]) == [TemperatureQuery('Outside_Temperature', 'outside')]
    with pytest.raises(ValueError):
        catalog.validate([{'name': 'get_inside_temperature', 'arguments': {'value': 30}}])


@pytest.mark.parametrize('state,unit,want', [('21.11111111111111 °C','°C','21.11 °C'), ('18.17','°C','18.17 °C'), ('NULL','°C','unavailable'), ('UNDEF','°C','unavailable'), ('0 °C','°C','0 °C')])
def test_query_reads_fresh_state_and_logs_result_on_same_line(tmp_path, monkeypatch, state, unit, want):
    api = OpenHAB('http://example.invalid')
    urls = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'state': state, 'unitSymbol': unit}
    monkeypatch.setattr(api.session, 'get', lambda url, **kw: urls.append(url) or Response())
    monkeypatch.setattr(api.session, 'post', lambda *a, **kw: pytest.fail('A temperature query must not POST'))
    transcript = observability.PlainTranscript(tmp_path / 'transcript.txt')
    monkeypatch.setattr(observability, 'plain_transcript', transcript)
    transcript.wake()
    transcript.update('1', 'Hey Jarvis what is the temperature inside?')
    assert execute_action(api, TemperatureQuery('Inside_Temperature', 'inside')) == want
    assert urls == ['http://example.invalid/rest/items/Inside_Temperature']
    text = transcript.path.read_text(encoding='utf-8')
    assert '\n' not in text
    assert json.loads(text[text.index('['):])[0]['result'] == want
