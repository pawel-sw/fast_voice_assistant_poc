from unittest.mock import Mock

import pytest

from jarvis.control import Catalog, OpenHAB, Action
from jarvis.integrations import HomeAssistant, create_integration


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv('HOMEASSISTANT_TOKEN', 'test-token')
    return HomeAssistant('http://ha.local/home/overview')


@pytest.mark.parametrize('config', [{}, {'openhab_url': '', 'homeassistant_url': None},
    {'openhab_url': 'http://oh', 'homeassistant_url': 'http://ha'}])
def test_exactly_one_backend(config):
    with pytest.raises(ValueError, match='exactly one'):
        create_integration(config)


def test_backend_selection(api):
    assert isinstance(create_integration({'openhab_url': 'http://oh'}), OpenHAB)
    assert isinstance(create_integration({'openhab_url': None, 'homeassistant_url': api.url}), HomeAssistant)
    assert api.url == 'http://ha.local'
    assert api.session.headers['Authorization'] == 'Bearer test-token'


def test_missing_token(monkeypatch):
    monkeypatch.delenv('HOMEASSISTANT_TOKEN', raising=False)
    with pytest.raises(ValueError, match='HOMEASSISTANT_TOKEN'):
        HomeAssistant('http://ha')


def test_discovery_and_catalog(api):
    api.session.get = Mock()
    api.session.get.return_value.json.return_value = [
        {'entity_id': 'light.kitchen', 'state': 'on', 'attributes': {
            'friendly_name': 'Kitchen Light', 'supported_color_modes': ['brightness']}},
        {'entity_id': 'light.porch', 'state': 'off', 'attributes': {'supported_color_modes': ['onoff']}},
        {'entity_id': 'switch.dead', 'state': 'unavailable'},
        {'entity_id': 'cover.shade', 'state': 'open', 'attributes': {'supported_features': 4}},
        {'entity_id': 'sensor.outside', 'state': '21', 'attributes': {'device_class': 'temperature'}}]
    catalog = Catalog(api.items(), {'temperature_items': {'outside': 'sensor.outside'}})
    schemas = {s['name'] for s in catalog.schemas}
    assert 'level_light_kitchen' in schemas
    assert 'level_light_porch' not in schemas
    assert 'switch_switch_dead' not in schemas
    assert 'position_cover_shade' in schemas and 'move_cover_shade' not in schemas
    assert 'get_outside_temperature' in schemas
    assert catalog.validate([{'name': 'switch_light_kitchen', 'arguments': {'value': 'ON'}}]) == [Action('light.kitchen', 'ON')]
    assert catalog.candidates('turn on kitchen light')
    api.session.get.assert_called_once_with('http://ha.local/api/states', timeout=15)


@pytest.mark.parametrize('entity,command,service,extra', [
    ('light.kitchen', 'ON', 'light/turn_on', {}),
    ('switch.fan', 'OFF', 'switch/turn_off', {}),
    ('input_boolean.test', 'ON', 'input_boolean/turn_on', {}),
    ('light.kitchen', '42', 'light/turn_on', {'brightness_pct': 42}),
    ('fan.bedroom', '60', 'fan/set_percentage', {'percentage': 60}),
    ('cover.shade', '30', 'cover/set_cover_position', {'position': 70}),
    ('cover.shade', 'UP', 'cover/open_cover', {}),
    ('cover.shade', 'DOWN', 'cover/close_cover', {}),
    ('cover.shade', 'STOP', 'cover/stop_cover', {}),
    ('media_player.tv', 'PLAY', 'media_player/media_play', {}),
    ('media_player.tv', 'NEXT', 'media_player/media_next_track', {})])
def test_service_commands(api, entity, command, service, extra):
    api.session.post = Mock()
    api.command(entity, command)
    api.session.post.assert_called_once_with('http://ha.local/api/services/' + service,
        json={'entity_id': entity, **extra}, timeout=8)
    api.session.post.return_value.raise_for_status.assert_called_once()


@pytest.mark.parametrize('entity,command', [('switch.test', '42'), ('light.test', '101'), ('lock.front', 'ON')])
def test_unsupported_command_does_not_send(api, entity, command):
    api.session.post = Mock()
    with pytest.raises(ValueError):
        api.command(entity, command)
    api.session.post.assert_not_called()


@pytest.mark.parametrize('state,expected', [('21.123', '21.12 °C'), ('unknown', 'unavailable'), ('unavailable', 'unavailable')])
def test_temperature(api, state, expected):
    api.session.get = Mock()
    api.session.get.return_value.json.return_value = {'state': state, 'attributes': {'unit_of_measurement': '°C'}}
    assert api.temperature('sensor.outside') == expected


def test_timer_and_http_failure(api):
    api.session.post = Mock()
    api.timer_state('00:42')
    assert api.session.post.call_args.args[0].endswith('/api/states/sensor.jarvis_timer_remaining')
    assert api.session.post.call_args.kwargs['json']['state'] == '00:42'
    api.session.post.return_value.raise_for_status.side_effect = RuntimeError('HTTP failure')
    with pytest.raises(RuntimeError):
        api.command('light.kitchen', 'ON')
