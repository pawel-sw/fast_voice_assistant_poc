from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from urllib.parse import quote

import requests

from .observability import timed, plain_wake, plain_command, plain_result, plain_candidate, plain_confirm_candidate, plain_cancel_candidate

log = logging.getLogger(__name__)


class OpenHAB:
    name = 'openhab'
    def __init__(self, url):
        self.url = url.rstrip('/')
        self.session = requests.Session()
        self.session.trust_env = False
        token = os.getenv('OPENHAB_TOKEN')
        if token:
            self.session.auth = (token, '')
        elif os.getenv('OPENHAB_PASSWORD'):
            self.session.auth = (os.getenv('OPENHAB_USER', 'admin'), os.environ['OPENHAB_PASSWORD'])

    def items(self):
        with timed('openhab.GET.items'):
            response = self.session.get(self.url + '/rest/items', params={'recursive': 'false'}, timeout=15)
            response.raise_for_status()
            return response.json()

    def command(self, item, command):
        # POST delivers a command to bindings/rules; PUT /state only updates the bus.
        with timed('openhab.POST.command', item=item, command=command):
            response = self.session.post(self.url + '/rest/items/' + quote(item, safe=''),
                                     data=command.encode(), headers={'Content-Type': 'text/plain'}, timeout=8)
            response.raise_for_status()
            return response.status_code

    def timer_state(self, value):
        from .timers import TIMER_ITEM
        with timed('openhab.PUT.timer', detail=True):
            response = self.session.put(self.url + '/rest/items/' + TIMER_ITEM + '/state',
                data=value.encode(), headers={'Content-Type': 'text/plain'}, timeout=(2, 3))
            response.raise_for_status()

    def temperature(self, item):
        with timed('openhab.GET.temperature', item=item):
            response = self.session.get(self.url + '/rest/items/' + quote(item, safe=''), timeout=8)
            response.raise_for_status()
            data = response.json()
            state = str(data.get('state', 'NULL')).strip()
            if state in ('NULL', 'UNDEF', ''):
                return 'unavailable'
            # Quantity states already carry their unit. Preserve it exactly.
            unit = data.get('unitSymbol')
            if unit and re.fullmatch(r'[-+]?\d+(?:\.\d+)?', state):
                state += ' ' + unit
            value = re.fullmatch(r'([-+]?\d+(?:\.\d+)?)\s*(.*)', state)
            if value:
                number = f'{float(value[1]):.2f}'.rstrip('0').rstrip('.')
                state = (number + ' ' + value[2]).strip()
            return state


@dataclass
class Action:
    item: str
    command: str


@dataclass
class TemperatureQuery:
    item: str
    location: str


def execute_action(api, action, dry_run=False):
    if isinstance(action, TemperatureQuery):
        try:
            value = api.temperature(action.item)
        except Exception as exc:
            plain_result(action.item, action.location, error=type(exc).__name__)
            raise
        plain_result(action.item, action.location, result=value)
        log.info('TEMPERATURE %s: %s (%s)', action.location, value, action.item)
        return value
    if dry_run:
        log.info('DRY RUN: %s <- %s', action.item, action.command)
        plain_command(action.item, action.command, dry_run=True)
        return
    try:
        status = api.command(action.item, action.command)
    except Exception as exc:
        plain_command(action.item, action.command, error=type(exc).__name__)
        raise
    plain_command(action.item, action.command)
    log.info('SENT HTTP %s: %s <- %s', status, action.item, action.command)


class Catalog:
    def __init__(self, items, config):
        from .item_policy import ItemPolicy
        self.policy = ItemPolicy(items, config)
        self.items = {}
        self.schemas = []
        self.actions = {}
        self.queries = {}
        available = {item['name']: item for item in items}
        for location, item in config.get('temperature_items', {'outside': 'Outside_Temperature', 'inside': 'Inside_Temperature'}).items():
            if item in available and not self.policy.excluded(available[item]):
                name = f'get_{location}_temperature'
                self.queries[name] = TemperatureQuery(item, location)
                self.schemas.append({'name': name, 'description': f'Read the current {location} temperature.',
                    'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}})
        self.aliases = {}
        include = set(config.get('include_items', []))
        for item in items:
            name, kind = item['name'], item['type']
            sd = item.get('stateDescription') or {}
            if self.policy.excluded(item) or sd.get('readOnly'):
                continue
            tags = set(item.get('tags', []))
            supported = kind in ('Switch', 'Dimmer', 'Color', 'Rollershutter', 'Player')
            control = bool(tags & {'Control', 'Switch', 'Blinds'}) or kind in ('Dimmer', 'Color', 'Rollershutter', 'Player')
            if not supported or not (control or name in include):
                continue
            self.items[name] = item
            explicit_aliases = config.get('aliases', {}).get(name, [])
            self.aliases[name] = list(dict.fromkeys(explicit_aliases +
                [self.policy.normalize(alias) for alias in explicit_aliases] + self.policy.aliases(item)))
            label = item.get('label') or name.replace('_', ' ')
            aliases = self.aliases[name]
            label += ('; also called ' + ', '.join(aliases)) if aliases else ''
            if kind in ('Switch', 'Dimmer', 'Color'):
                self.add(name, 'switch', f'Turn {label} on or off.', {'type': 'string', 'enum': ['ON', 'OFF']})
            if kind in ('Dimmer', 'Color'):
                self.add(name, 'level', f'Set {label} brightness, volume or fan level to a percentage.', {'type': 'integer', 'minimum': 0, 'maximum': 100})
            if kind == 'Rollershutter':
                self.add(name, 'move', f'Open, close, or stop {label}.', {'type': 'string', 'enum': ['open', 'close', 'stop']})
                self.add(name, 'position', f'Set {label} closure percentage: 0 fully open, 100 closed.', {'type': 'integer', 'minimum': 0, 'maximum': 100})
            if kind == 'Player':
                self.add(name, 'play', f'Control playback of {label}.', {'type': 'string', 'enum': ['PLAY', 'PAUSE', 'NEXT', 'PREVIOUS']})

    def add(self, item, action, description, value):
        if 'actions' in self.items[item] and action not in self.items[item]['actions']:
            return
        name = action + '_' + item.lower().replace('.', '_')
        self.actions[name] = (item, value)
        self.schemas.append({'name': name, 'description': description, 'parameters': {
            'type': 'object', 'properties': {'value': value}, 'required': ['value'], 'additionalProperties': False}})

    def validate(self, calls):
        if not isinstance(calls, list) or not calls or len(calls) > 12:
            raise ValueError('No valid tool calls')
        result = []
        seen = set()
        for call in calls:
            if isinstance(call, dict) and call.get('name') in self.queries:
                if call.get('arguments') != {}:
                    raise ValueError('Temperature queries take no arguments')
                if call['name'] not in seen:
                    result.append(self.queries[call['name']])
                    seen.add(call['name'])
                continue
            if not isinstance(call, dict) or call.get('name') not in self.actions:
                raise ValueError('Unknown tool')
            item, schema = self.actions[call['name']]
            args = call.get('arguments')
            if not isinstance(args, dict) or set(args) != {'value'}:
                raise ValueError('Invalid arguments')
            value = args['value']
            if 'enum' in schema:
                if value not in schema['enum']:
                    raise ValueError('Invalid command')
            elif type(value) is not int or not 0 <= value <= 100:
                raise ValueError('Invalid percentage')
            key = (item, str(value))
            if call['name'].startswith('move_'):
                key = (item, {'open': 'UP', 'close': 'DOWN', 'stop': 'STOP'}[value])
            if key not in seen:
                result.append(Action(*key))
                seen.add(key)
        return result

    def candidates(self, text):
        if self.policy.ignored_request(text):
            return []
        text = self.policy.normalize(text)
        def words(value):
            value = value.lower().replace('_', ' ')
            synonyms = {'lights': 'light', 'lamps': 'lamp', 'shades': 'shade', 'blinds': 'shade', 'theater': 'theatre'}
            return {synonyms.get(w, w) for w in re.findall(r'[a-z0-9]+', value)}
        query = words(text)
        temperature_schemas = []
        if query & {'temperature', 'temperatures', 'temp', 'hot', 'cold', 'warm'}:
            for schema in self.schemas:
                if schema['name'] not in self.queries:
                    continue
                location = self.queries[schema['name']].location
                aliases = {'outside', 'outdoor', 'outdoors'} if location == 'outside' else {'inside', 'indoor', 'indoors'}
                if query & aliases:
                    temperature_schemas.append(schema)
        matches = []
        for name, item in self.items.items():
            labels = [item.get('label') or name, name, *self.aliases.get(name, [])]
            for label in labels:
                if not self.policy.label_matches_room(label, text):
                    continue
                target = words(label) - {'rgb', 'dimmer', 'toggle', 'color'}
                # Require the complete named target, not a model's guessed room.
                if target and target <= query:
                    matches.append((frozenset(target), name))
        if not matches:
            return temperature_schemas
        names = {name for _, name in matches}
        # A longer matching name takes precedence over its contained short name.
        for name in list(names):
            own = [target for target, item_name in matches if item_name == name]
            if all(any(target < other_target and other_name != name for other_target, other_name in matches) for target in own):
                names.discard(name)
        numeric = bool(re.search(r'\b(dim|percent|percentage|brightness|level|speed)\b|\b(?:to|at)\s+\d+', text, re.I))
        return temperature_schemas + [schema for schema in self.schemas if schema['name'] in self.actions and self.actions[schema['name']][0] in names
                and (schema['name'].startswith(('level_', 'position_')) == numeric)]


class Planner:
    def __init__(self, catalog, min_confidence=0.8):
        os.environ['NEEDLE_TELEMETRY'] = '0'
        from needle import Needle
        self.catalog = catalog
        self.minimum = min_confidence
        self.factory = Needle
        self.model = None

    def plan(self, text):
        text = ' '.join(text.split())
        if not text:
            return []
        if self.catalog.policy.ignored_request(text):
            return []
        text = self.catalog.policy.normalize(text)
        # Needle's confidence is sensitive to a missing sentence boundary.
        # Keep ASR punctuation; give unpunctuated complete requests the same form.
        if text[-1] not in '.!?':
            text += '.'
        with timed('catalog.select'):
            schemas = self.catalog.candidates(text)
        if not schemas:
            log.info('Awaiting a complete named device')
            return []
        if self.model is not None:
            with timed('needle.close'):
                self.model.close()
        with timed('needle.load', tools=len(schemas)):
            self.model = self.factory(generation=3, tools=schemas, auto_date=False,
                                 system='Map explicit home actions and temperature queries to tools. Incomplete, negated or unsupported requests return no calls.')
        with timed('needle.complete'):
            response = self.model.complete(text, max_new_tokens=256)
        log.info('Needle: %s', json.dumps(response))
        confidence = response.get('confidence')
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or confidence < self.minimum:
            log.info('Rejected Needle confidence %s; minimum %.2f', confidence, self.minimum)
            return []
        try:
            calls = response.get('function_calls')
            if any(c.get('name') not in {s['name'] for s in schemas} for c in calls or []):
                return []
            with timed('tools.validate'):
                actions = self.catalog.validate(calls)
                for call in calls:
                    if call['name'].startswith('switch_'):
                        value = call['arguments']['value']
                        if not re.search(r'\b' + re.escape(value) + r'\b', text, re.I):
                            raise ValueError(f'Switch command {value} was not explicitly spoken')
                return actions
        except ValueError as exc:
            log.info('Awaiting continuation: %s', exc)
            return []


class Conversation:
    """One wake session; rejected fragments accumulate, accepted actions close it."""
    def __init__(self, planner, execute, wake='hey jarvis', continuation=5.0):
        self.planner, self.execute = planner, execute
        pattern = r'(?:hey[\s,!.?]+jarvis|alexa)' if wake.lower() == 'hey jarvis' else r'[\s,!.?]+'.join(map(re.escape, wake.split()))
        self.wake = re.compile(r'\b' + pattern + r'\b', re.I)
        self.continuation = continuation
        self.pending = None
        self.deadline = 0.0
        self.verified = True

    def expire(self, now):
        if self.pending is not None and now > self.deadline:
            log.info('Wake session expired without a valid command')
            self.pending = None
            plain_cancel_candidate()

    def activate(self, now, tentative=False):
        """Only the acoustic wake detector may open a listening session."""
        self.verified = not tentative
        if tentative:
            plain_candidate()
        else:
            plain_cancel_candidate()
            plain_wake()
        self.pending = ''
        self.deadline = now + self.continuation

    def confirm_wake(self, text):
        if not self.verified and self.wake.search(text):
            self.verified = True
            plain_confirm_candidate()
            log.info('Wake candidate confirmed by R2T2')

    def utterance(self, text, now):
        self.expire(now)
        if self.pending is not None and not self.verified:
            self.confirm_wake(text)
            if not self.verified:
                self.pending = None
                plain_cancel_candidate()
                log.info('Wake candidate rejected: Hey Jarvis or Alexa was not transcribed')
                return []
        if not text.strip():
            return []
        if self.pending is None:
            return []
        match = self.wake.search(text)
        if match:
            self.pending = text[match.end():].lstrip(' ,.!?').rstrip()
            log.debug('Removed wake phrase from command transcript')
        elif self.pending is not None:
            self.pending = (self.pending + ' ' + text).strip()
        else:
            return []
        self.deadline = now + self.continuation
        if not self.pending:
            return []
        actions = self.planner.plan(self.pending)
        if actions:
            # Close before sending: a timeout may mean the command already arrived.
            self.pending = None
            for action in actions:
                self.execute(action)
        return actions
