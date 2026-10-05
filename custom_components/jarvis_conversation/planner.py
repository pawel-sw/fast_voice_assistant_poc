"""Async inference only. This module cannot execute home actions."""
import asyncio
from dataclasses import dataclass, field
import json
import math
import logging
import re
from urllib.parse import urlsplit, urlunsplit
import uuid

import aiohttp

from .const import DEFAULT_MODEL, PROMPT

LOGGER = logging.getLogger(__name__)


def select_tools(text, schemas):
    """Keep a small relevant catalog for Needle and the cloud token budget."""
    words = set(re.findall(r'[a-z]+', text.lower()))
    read_only = not bool(words & {'turn', 'set', 'switch', 'open', 'close', 'start', 'stop', 'pause', 'play', 'lock', 'unlock', 'on', 'off', 'dim', 'increase', 'decrease', 'add', 'remove', 'cancel', 'next', 'previous'})
    if words & {'what', 'which', 'is', 'are', 'how', 'when', 'check'} and not words & {'turn', 'set', 'switch', 'open', 'close', 'start', 'stop', 'pause', 'play', 'lock', 'unlock'}:
        read_only = True
    if words & {'if', 'unless'}:
        read_only = True  # Never run a conditional write without evaluating its condition.
    desired = ['getlivecontext']
    mappings = {
        'on': ['turnon'], 'off': ['turnoff'], 'brightness': ['lightset'],
        'dim': ['lightset'], 'color': ['lightset'], 'temperature': ['climatesettemperature'],
        'open': ['turnon', 'cover'], 'close': ['turnoff', 'cover'],
        'pause': ['mediapause'], 'play': ['mediaunpause', 'mediasearch'],
        'next': ['medianext'], 'previous': ['mediaprevious'],
        'volume': ['volume'], 'timer': ['timer'], 'weather': ['weather'],
        'lock': ['lock'], 'unlock': ['unlock'], 'list': ['list'],
    }
    for word in words:
        desired.extend(mappings.get(word, []))
    def score(schema):
        name = schema['name'].lower()
        return sum(10 for token in desired if token in name) + len(words & set(re.findall(r'[a-z]+', schema.get('description', '').lower())))
    candidates = [s for s in schemas if not read_only or any(token in s['name'].lower() for token in ('get', 'search'))]
    return sorted(candidates, key=score, reverse=True)[:10]


def rpc_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ('ws', 'wss') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Use a ws:// or wss:// backend URL without embedded credentials')
    return urlunsplit((parsed.scheme, parsed.netloc, '/rpc', '', ''))


@dataclass
class Plan:
    calls: list = field(default_factory=list)
    reply: str = ''
    stage: str = 'none'


def validate_plan(response, schemas, validate_args, stage, minimum=0.8):
    """Reject the whole proposal if any call is malformed or unrecognized."""
    if not isinstance(response, dict):
        return None
    confidence = response.get('confidence')
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not minimum <= confidence <= 1:
        return None
    calls = response.get('function_calls')
    if not isinstance(calls, list) or len(calls) > 8:
        return None
    allowed = {schema['name'] for schema in schemas}
    result, seen = [], set()
    for call in calls:
        if not isinstance(call, dict) or call.get('name') not in allowed or not isinstance(call.get('arguments'), dict):
            return None
        try:
            args = validate_args(call['name'], call['arguments'])
            key = json.dumps([call['name'], args], sort_keys=True)
        except (ValueError, TypeError, KeyError):
            LOGGER.debug('Tool argument validation rejected %s', call['name'])
            return None
        if key in seen:
            return None
        seen.add(key)
        result.append({'name': call['name'], 'arguments': args})
    reply = response.get('reply', '')
    if not isinstance(reply, str):
        return None
    # Only Groq can provide a conversational answer without tool calls.
    if not result and (stage != 'groq' or not reply.strip()):
        return None
    return Plan(result, reply.strip()[:900], stage)


class Planner:
    def __init__(self, session, settings):
        self.session = session
        self.settings = settings

    async def needle(self, text, schemas, prompt):
        async with asyncio.timeout(18):
            async with self.session.ws_connect(rpc_url(self.settings['backend_url']),
                headers={'Authorization': 'Bearer ' + self.settings['backend_token']},
                max_msg_size=128000, heartbeat=10) as ws:
                ready = await ws.receive_json()
                if ready.get('type') != 'ready' or ready.get('protocol') != 2:
                    raise ValueError('Incompatible inference worker')
                await ws.send_json({'type': 'plan', 'session': uuid.uuid4().hex,
                    'text': text, 'schemas': schemas, 'system': prompt})
                async for event in ws:
                    if event.type != aiohttp.WSMsgType.TEXT:
                        raise ValueError('Inference worker disconnected')
                    data = json.loads(event.data)
                    if data.get('type') == 'plan':
                        return data['response']
                    if data.get('type') == 'error':
                        raise ValueError('Inference worker rejected request')
                raise ValueError('No inference response')

    async def groq(self, instruction, payload):
        body = {'model': self.settings.get('groq_model', DEFAULT_MODEL),
            'temperature': 0, 'max_completion_tokens': 1536,
            'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': instruction},
                         {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}
        if body['model'] == DEFAULT_MODEL:
            body['reasoning_effort'] = 'low'
        async with self.session.post('https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': 'Bearer ' + self.settings['groq_api_key']},
            json=body, timeout=aiohttp.ClientTimeout(total=18)) as response:
            if response.status != 200:
                LOGGER.warning('Groq fallback HTTP status=%d', response.status)
                raise ValueError(f'Groq HTTP {response.status}')
            data = await response.json()
        choice = data['choices'][0]
        if choice.get('finish_reason') != 'stop':
            LOGGER.warning('Groq fallback incomplete: %s', choice.get('finish_reason'))
            raise ValueError('Incomplete Groq response')
        return json.loads(choice['message']['content'])

    async def plan(self, text, schemas, context, history, validate_args):
        """Fallback completes before any action is executed; cancellation propagates."""
        if not text.strip() or len(text) > 3000:
            return Plan(reply='Please use a shorter request.')
        if len(schemas) > 100:
            return Plan(reply='The Assist tool catalog is unavailable or too large.')
        minimum = self.settings.get('min_confidence', 0.8)
        def checked(value, stage):
            result = validate_plan(value, schemas, validate_args, stage, minimum)
            if not result:
                LOGGER.debug('Rejected stage=%s confidence=%s call_names=%s', stage,
                    value.get('confidence') if isinstance(value, dict) else None,
                    [c.get('name') for c in value.get('function_calls', []) if isinstance(c, dict)]
                    if isinstance(value, dict) and isinstance(value.get('function_calls'), list) else [])
            return result
        # The existing worker limits the system prompt to 4000 characters.
        needle_prompt = PROMPT + '\nRelevant Home Assistant context:\n' + context[:2200]
        if history:
            needle_prompt += '\nRecent conversation: ' + json.dumps(history, ensure_ascii=False)[-1000:]
        needle_available = True
        try:
            if not schemas:
                raise ValueError('No applicable local tools')
            result = checked(await self.needle(text, schemas, needle_prompt), 'needle')
            if result:
                return result
        except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, TypeError):
            needle_available = False
        if not self.settings.get('groq_api_key'):
            return Plan(reply='I could not match that request. Please name the device and action clearly.')
        payload = {'transcript': text, 'home_context': context, 'recent_conversation': history}
        corrected = text
        try:
            repair = await self.groq(PROMPT + ' Repair phonetic transcription errors only. '
                'Preserve intent and negation; if uncertain, keep the original. '
                'Return JSON with one string field: corrected.', payload)
            value = repair.get('corrected')
            if isinstance(value, str) and value.strip() and len(value) <= 3000 and '\n' not in value:
                corrected = value.strip()
        except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, TypeError):
            pass
        if needle_available:
            try:
                result = checked(await self.needle(corrected, schemas, needle_prompt), 'needle-retry')
                if result:
                    return result
            except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, TypeError):
                pass
        try:
            proposal = await self.groq(PROMPT + ' Return a JSON object with confidence (0 to 1), '
                'function_calls (list of objects with name and arguments), and reply (a short string). '
                'Use function_calls for home actions or live state queries. For ambiguity ask a '
                'clarifying question in reply with no calls. For general questions answer in reply. '
                'Never invent current device states. Do not claim execution in reply.',
                {**payload, 'corrected_transcript': corrected, 'available_tools': schemas})
            result = checked(proposal, 'groq')
            if result:
                return result
        except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, TypeError):
            pass
        return Plan(reply='I could not safely match that request. Please try rephrasing it.')


def result_speech(result):
    """Prefer HA's own intent response; never describe failure as success."""
    if not isinstance(result, dict):
        return 'The request returned an unexpected response.', True
    if result.get('error') or result.get('response_type') == 'error':
        return 'I could not confirm that request. Please check the device.', True
    data = result.get('data', result)
    if isinstance(data, dict) and data.get('failed'):
        return 'Some requested devices could not be controlled. Please check them.', True
    speech = result.get('speech', {})
    if isinstance(speech, dict):
        plain = speech.get('plain', {})
        if isinstance(plain, dict) and plain.get('speech'):
            return str(plain['speech'])[:900], False
    if isinstance(data, dict) and isinstance(data.get('success'), list) and data['success']:
        names = [str(item.get('name', item.get('id', 'device'))) for item in data['success'] if isinstance(item, dict)]
        return 'Done' + (': ' + ', '.join(names) if names else '') + '.', False
    if result.get('success') is True and isinstance(result.get('result'), str):
        # HA's GetLiveContext returns a fixed heading followed by YAML entities.
        import yaml
        raw = result['result']
        if raw.startswith('Live Context:') and '\n' in raw:
            entities = yaml.safe_load(raw.split('\n', 1)[1])
            if isinstance(entities, list):
                replies = []
                for entity in entities[:8]:
                    if not isinstance(entity, dict):
                        continue
                    attrs = entity.get('attributes', {})
                    unit = attrs.get('unit_of_measurement', '')
                    replies.append(f"{entity.get('names', 'Device')}: {entity.get('state', 'unknown')} {unit}".strip())
                if replies:
                    return '. '.join(replies)[:900], False
        return raw[:900], False
    # Query tools may return structured values rather than an intent speech field.
    return json.dumps(data, ensure_ascii=False)[:900], False
