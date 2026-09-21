"""Select one home automation backend and adapt Home Assistant entities."""
import os
import re
from urllib.parse import quote, urlsplit, urlunsplit

import requests

from .control import OpenHAB


def create_integration(config):
    selected = [(key, config.get(key)) for key in ('openhab_url', 'homeassistant_url')
                if isinstance(config.get(key), str) and config[key].strip()]
    if len(selected) != 1:
        raise ValueError('Configure exactly one of openhab_url or homeassistant_url')
    key, url = selected[0]
    return HomeAssistant(url) if key == 'homeassistant_url' else OpenHAB(url.strip())


class HomeAssistant:
    name = 'homeassistant'

    def __init__(self, url):
        parts = urlsplit(url.strip())
        if parts.scheme not in ('http', 'https') or not parts.netloc:
            raise ValueError('homeassistant_url must be an HTTP or HTTPS URL')
        path = parts.path.rstrip('/')
        if path == '/home/overview':
            path = ''
        self.url = urlunsplit((parts.scheme, parts.netloc, path, '', ''))
        token = os.getenv('HOMEASSISTANT_TOKEN', '').strip()
        if not token:
            raise ValueError('Set HOMEASSISTANT_TOKEN in the client .env')
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({'Authorization': 'Bearer ' + token})

    def items(self):
        response = self.session.get(self.url + '/api/states', timeout=15)
        response.raise_for_status()
        items = []
        for entity in response.json():
            name = entity['entity_id']
            domain = name.split('.')[0]
            attrs = entity.get('attributes', {})
            features = attrs.get('supported_features', 0)
            actions = []
            kind = 'String'
            if domain in ('switch', 'input_boolean', 'light', 'fan'):
                kind, actions = 'Switch', ['switch']
                modes = set(attrs.get('supported_color_modes', []))
                if (domain == 'light' and modes - {'onoff', 'unknown'}) or (domain == 'fan' and features & 1):
                    kind, actions = 'Dimmer', ['switch', 'level']
            elif domain == 'cover':
                kind = 'Rollershutter'
                if features & 1 and features & 2 and features & 8:
                    actions.append('move')
                if features & 4:
                    actions.append('position')
            elif domain == 'media_player':
                kind = 'Player'
                if all(features & flag for flag in (1, 16, 32, 16384)):
                    actions.append('play')
            elif domain == 'sensor' and attrs.get('device_class') == 'temperature':
                kind = 'Number:Temperature'
            items.append({'name': name, 'label': attrs.get('friendly_name', name),
                          'type': kind, 'state': entity.get('state'),
                          'tags': ['Control'] if actions else [], 'actions': actions,
                          'stateDescription': {'readOnly': entity.get('state') in ('unavailable', 'unknown') or not actions}})
        return items

    def command(self, item, command):
        domain = item.split('.')[0]
        data = {'entity_id': item}
        service = None
        if domain in ('light', 'switch', 'input_boolean', 'fan') and command in ('ON', 'OFF'):
            service = 'turn_on' if command == 'ON' else 'turn_off'
        elif domain == 'cover' and command in ('UP', 'DOWN', 'STOP'):
            service = {'UP': 'open_cover', 'DOWN': 'close_cover', 'STOP': 'stop_cover'}[command]
        elif domain == 'media_player':
            service = {'PLAY': 'media_play', 'PAUSE': 'media_pause', 'NEXT': 'media_next_track',
                       'PREVIOUS': 'media_previous_track'}.get(command)
        elif re.fullmatch(r'\d{1,3}', command) and 0 <= int(command) <= 100:
            value = int(command)
            if domain == 'light':
                service, data['brightness_pct'] = 'turn_on', value
            elif domain == 'fan':
                service, data['percentage'] = 'set_percentage', value
            elif domain == 'cover':
                service, data['position'] = 'set_cover_position', 100 - value
        if service is None:
            raise ValueError(f'Unsupported Home Assistant command: {item} {command}')
        response = self.session.post(self.url + f'/api/services/{domain}/{service}', json=data, timeout=8)
        response.raise_for_status()
        return response.status_code

    def temperature(self, item):
        response = self.session.get(self.url + '/api/states/' + quote(item, safe=''), timeout=8)
        response.raise_for_status()
        entity = response.json()
        state = str(entity.get('state', 'unavailable'))
        if state in ('unknown', 'unavailable', ''):
            return 'unavailable'
        if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', state):
            state = f'{float(state):.2f}'.rstrip('0').rstrip('.')
        unit = entity.get('attributes', {}).get('unit_of_measurement', '')
        return f'{state} {unit}'.strip()

    def timer_state(self, value):
        response = self.session.post(self.url + '/api/states/sensor.jarvis_timer_remaining',
            json={'state': value, 'attributes': {'friendly_name': 'Jarvis Timer Remaining'}}, timeout=(2, 3))
        response.raise_for_status()
