from datetime import datetime, timezone, timedelta
from unittest.mock import Mock
import pytest
from jarvis.weather import WeatherQuery, WeatherService, parse_weather
from jarvis.remote_rpc import RemotePlanner
from test_client_brain import CATALOG, make_brain


@pytest.mark.parametrize('text,period', [
    ("what's the weather?", 'current'), ('weather in London UK', 'current'),
    ('what is the temperature in London', 'current'),
    ('tell me the weather forecast today', 'today'), ('weather tomorrow', 'tomorrow'),
    ('weather in Vancouver', 'unsupported'), ('weather next week', 'unsupported')])
def test_weather_intents(text, period):
    assert parse_weather(text, {'weather': {'name': 'London', 'aliases': ['UK']}}) == WeatherQuery(period)


def test_sensor_and_device_requests_are_not_weather():
    assert parse_weather('what is the temperature outside') is None
    assert parse_weather('turn on weather light') is None


def data():
    return {'utc_offset_seconds': -25200, 'current': {'time': '2026-09-20T12:00',
        'temperature_2m': 18, 'apparent_temperature': 16, 'wind_speed_10m': 9, 'weather_code': 2},
        'daily': {'time': ['2026-09-20', '2026-09-21'], 'weather_code': [2, 61],
            'temperature_2m_max': [21, 19], 'temperature_2m_min': [9, 8],
            'precipitation_probability_max': [10, 70]}}


def service():
    now = datetime(2026, 9, 20, 12, 10, tzinfo=timezone(timedelta(hours=-7))).timestamp()
    return WeatherService({'weather': {'name': 'London'}}, now=lambda: now)


def test_current_and_tomorrow_have_correct_day_and_units():
    current = service().format(data(), 'current')
    assert '18 degrees Celsius' in current and 'partly cloudy' in current
    assert '9 kilometres per hour' in current and '10 percent' in current
    tomorrow = service().format(data(), 'tomorrow')
    assert 'Tomorrow in London' in tomorrow and 'light rain' in tomorrow
    assert 'high of 19' in tomorrow and '70 percent' in tomorrow
    assert 'currently' not in tomorrow


def test_stale_or_missing_weather_is_not_fabricated():
    payload = data()
    payload['current']['time'] = '2026-09-19T12:00'
    with pytest.raises(ValueError, match='stale'):
        service().format(payload, 'current')
    payload = data()
    payload['current']['temperature_2m'] = None
    with pytest.raises(ValueError):
        service().format(payload, 'current')


def test_client_reads_weather_and_logs_then_speaks(tmp_path, monkeypatch):
    from jarvis import observability as obs
    path = tmp_path/'live.txt'
    monkeypatch.setattr(obs, 'plain_transcript', obs.PlainTranscript(path))
    brain, api, speaker = make_brain(action=WeatherQuery())
    brain.weather = Mock()
    brain.weather.answer.return_value = 'In London, it is 18 degrees Celsius.'
    session = brain.start()
    brain.transcript({'session': session, 'utterance': 0, 'text': 'Alexa what is the weather', 'final': True})
    brain.tick()
    assert speaker.replies == ['In London, it is 18 degrees Celsius.']
    assert '[weather: In London, it is 18 degrees Celsius.]' in path.read_text()
    assert not api.calls


def test_unavailable_weather_has_spoken_error():
    brain, api, speaker = make_brain(action=WeatherQuery())
    brain.weather = Mock()
    brain.weather.answer.side_effect = TimeoutError()
    session = brain.start()
    brain.transcript({'session': session, 'utterance': 0, 'text': 'weather', 'final': True})
    brain.tick()
    assert 'could not get the weather' in speaker.replies[0]
    assert not api.calls


def test_weather_does_not_need_needle_or_groq():
    p = RemotePlanner(CATALOG, {'min_confidence': .8})
    p._needle = Mock(side_effect=AssertionError('No inference for weather routing'))
    assert p.plan('weather tomorrow', 's') == [WeatherQuery('tomorrow')]
