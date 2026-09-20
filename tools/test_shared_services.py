"""Real concurrent VM smoke test, without home control or speaker playback."""
from concurrent.futures import ThreadPoolExecutor
import json
from math import gcd
import os
from pathlib import Path
import sys
import threading
import time
import wave
import numpy as np
from scipy.signal import resample_poly
from dotenv import load_dotenv
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'clients'))
from inference_client import InferenceClient

load_dotenv(ROOT / '.env')
config = json.loads((ROOT / 'config.json').read_text())
client = InferenceClient(config['remote_url'])
http_url = config['remote_url'].replace('wss://', 'https://', 1).replace('ws://', 'http://', 1).rstrip('/')
headers = {'Authorization': 'Bearer ' + os.environ['JARVIS_REMOTE_TOKEN']}
response = requests.get(http_url + '/services', headers=headers, timeout=10)
response.raise_for_status()
services = response.json()
assert services['tts']['speed'] == 1.2
assert requests.get(http_url + '/services', timeout=10).status_code == 401
barrier = threading.Barrier(5)


def recognize(label):
    with wave.open(str(ROOT / f'data/test-{label}.wav')) as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32)
    if rate != 16000:
        factor = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // factor, rate // factor)
    pcm = bytes(16000) + np.clip(samples, -32768, 32767).astype('<i2').tobytes() + bytes(24000)
    barrier.wait(timeout=10)
    def chunks():
        for offset in range(0, len(pcm), 15360):
            yield pcm[offset:offset + 15360]
            time.sleep(.48)
    events = list(client.transcribe(chunks()))
    final = events[-1]
    assert final['final'] and label in final['text'].lower(), final
    other = 'inside' if label == 'outside' else 'outside'
    assert other not in final['text'].lower(), final
    return {'asr': label, 'text': final['text']}


def speak(speed):
    barrier.wait(timeout=10)
    pcm = b''.join(client.speak('The temperature outside is twelve degrees Celsius. Your export is ready.', speed=speed))
    assert len(pcm) > 24000
    return {'speed': speed or 1.2, 'seconds': len(pcm) / 48000}


def plan():
    barrier.wait(timeout=10)
    result = client.plan('Search for blue shoes.', [{'name': 'search_catalog',
        'description': 'Search the product catalog', 'parameters': {'type': 'object',
            'properties': {'query': {'type': 'string'}}, 'required': ['query']}}],
        system='Select the appropriate catalog tool for the request.')
    assert result.get('function_calls'), result
    assert result['function_calls'][0]['name'] == 'search_catalog', result
    return {'plan': result}


if __name__ == '__main__':
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(recognize, 'outside'), pool.submit(recognize, 'inside'),
            pool.submit(speak, 1.0), pool.submit(speak, None), pool.submit(plan)]
        results = [future.result(timeout=90) for future in futures]
    assert results[3]['seconds'] < results[2]['seconds'] * .92, results
    print(json.dumps({'concurrent_requests': 5, 'elapsed': round(time.monotonic()-started, 2),
        'results': results}, indent=2))
