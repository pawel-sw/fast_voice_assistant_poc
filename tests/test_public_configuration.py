from unittest.mock import Mock

from jarvis.control import Catalog, TemperatureQuery
from jarvis.weather import WeatherQuery, WeatherService, parse_weather


def test_custom_temperature_item_is_used():
    catalog = Catalog([{'name': 'Garden_Sensor', 'type': 'Number:Temperature'}],
                      {'temperature_items': {'outside': 'Garden_Sensor'}})
    assert catalog.validate([{'name': 'get_outside_temperature', 'arguments': {}}]) == [
        TemperatureQuery('Garden_Sensor', 'outside')]
    assert not catalog.candidates('temperature inside')


def test_weather_uses_configured_location(monkeypatch):
    config = {'weather': {'name': 'Example City', 'aliases': ['Exampleville'],
                         'latitude': 42.5, 'longitude': -71.2, 'timezone': 'UTC'}}
    assert parse_weather('temperature in Exampleville', config) == WeatherQuery()
    assert parse_weather('weather in another city', config) == WeatherQuery('unsupported')
    fetch = Mock()
    fetch.return_value.json.return_value = {'fixture': True}
    monkeypatch.setattr('jarvis.weather.requests.get', fetch)
    service = WeatherService(config)
    service.format = Mock(return_value='Weather reply')
    assert service.answer(WeatherQuery()) == 'Weather reply'
    params = fetch.call_args.kwargs['params']
    assert (params['latitude'], params['longitude'], params['timezone']) == (42.5, -71.2, 'UTC')


def test_unconfigured_weather_does_not_fetch(monkeypatch):
    fetch = Mock(side_effect=AssertionError('No implicit private location'))
    monkeypatch.setattr('jarvis.weather.requests.get', fetch)
    assert 'configure' in WeatherService({}).answer(WeatherQuery())


def test_default_and_index_microphones(monkeypatch):
    from jarvis import remote_client
    sd = Mock()
    sd.default.device = (4, 5)
    sd.query_devices.return_value = {'name': 'Test microphone'}
    monkeypatch.setattr(remote_client, 'sd', sd)
    assert remote_client.microphone(None)[0] == 4
    assert remote_client.microphone(7)[0] == 7
    sd.check_input_settings.assert_called_with(device=7, channels=1, samplerate=16000, dtype='int16')
