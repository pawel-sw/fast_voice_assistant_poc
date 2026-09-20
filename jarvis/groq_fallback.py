"""Client-only transcript repair and tool proposals using Groq."""
import json
from pathlib import Path
import requests
from .observability import timed

ROOT = Path(__file__).resolve().parents[1]


class GroqFallback:
    def __init__(self, config, catalog):
        self.token = config.get('groq_token', '')
        self.catalog = catalog

    def _complete(self, prompt_file, payload, stage, json_output=False):
        body = {'model': 'openai/gpt-oss-120b', 'temperature': 0,
            'reasoning_effort': 'low', 'max_completion_tokens': 1024,
            'messages': [
                {'role': 'system', 'content': (ROOT / prompt_file).read_text(encoding='utf-8')},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}
        if json_output:
            body['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'openhab_proposal', 'strict': True, 'schema': {
                    'type': 'object', 'additionalProperties': False,
                    'required': ['confidence', 'function_calls'], 'properties': {
                        'confidence': {'type': 'number'},
                        'function_calls': {'type': 'array', 'items': {
                            'type': 'object', 'additionalProperties': False,
                            'required': ['name', 'arguments'], 'properties': {
                                'name': {'type': 'string'},
                                'arguments': {'anyOf': [
                                    {'type': 'object', 'properties': {}, 'additionalProperties': False},
                                    {'type': 'object', 'properties': {'value': {'type': ['string', 'integer']}}, 'required': ['value'], 'additionalProperties': False}
                                ]}}}}}}}}
        with timed('groq.' + stage):
            response = requests.post('https://api.groq.com/openai/v1/chat/completions',
                headers={'Authorization': 'Bearer ' + self.token}, json=body, timeout=(5, 25))
            # Do not include provider bodies, headers or credentials in errors/logs.
            if not response.ok:
                raise RuntimeError(f'Groq {stage} HTTP {response.status_code}')
            data = response.json()
            choice = data['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Groq response incomplete')
            content = choice['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise ValueError('Groq returned no content')
            return content.strip()

    def repair(self, text):
        devices = [{'label': item.get('label') or name,
                    'aliases': self.catalog.aliases.get(name, [])}
                   for name, item in self.catalog.items.items()]
        result = self._complete('SYSTEM_PROMPT_FIX_PHONETIC.md',
            {'transcript': text, 'known_devices': devices,
             'room_mappings': self.catalog.policy.mappings, 'room_defaults': self.catalog.policy.defaults,
             'item_defaults': self.catalog.policy.item_defaults,
             'temperature_locations': [q.location for q in self.catalog.queries.values()]}, 'repair')
        if len(result) > 4000 or '\n' in result:
            raise ValueError('Invalid repaired transcript')
        return result

    def propose(self, original, corrected):
        # Validation uses this same shortlist; avoid sending unrelated room tools.
        schemas = self.catalog.candidates(corrected) or self.catalog.schemas
        content = self._complete('SYSTEM_PROMPT_OPENHAB.md',
            {'original_transcript': original, 'corrected_transcript': corrected,
             'available_tools': schemas}, 'tools', json_output=True)
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError('Invalid Groq tool response')
        return result
