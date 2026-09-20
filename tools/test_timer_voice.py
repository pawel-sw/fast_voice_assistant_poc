"""Exercise real wake/ASR, timer expiry and wakeless stop; no device commands."""
from pathlib import Path
from math import gcd
import json
import sys
import threading
import time
import wave
from unittest.mock import patch
import numpy as np
from scipy.signal import resample_poly
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jarvis import remote_client as client
from jarvis.control import OpenHAB
from jarvis.timers import TimerService, TIMER_ITEM
from jarvis.remote_rpc import SpeechOutput
from jarvis.observability import configure_plain_transcript


def load(path):
    with wave.open(str(path)) as source:
        rate = source.getframerate()
        data = np.frombuffer(source.readframes(source.getnframes()), dtype='<i2').astype(np.float32)
    if rate != 16000:
        factor = gcd(rate, 16000)
        data = resample_poly(data, 16000//factor, rate//factor)
    return np.clip(data, -32768, 32767).astype('<i2').tobytes()


load_dotenv(ROOT/'.env')
config = json.loads((ROOT/'config.json').read_text())
config['timer_state_path'] = str(ROOT/'data/test-timer-voice-state.json')
Path(config['timer_state_path']).write_text('{"deadline":null,"ringing":false}')
pcm = bytes(32000*3) + load(ROOT/'data/timer-start.wav') + bytes(32000*12) + load(ROOT/'data/timer-stop.wav') + bytes(32000*10)
pcm += bytes(-len(pcm) % 640)
spoken = bytearray()
chimes = []
states = []
api = OpenHAB(config['openhab_url'])


class Input:
    def __init__(self, **kwargs):
        self.callback = kwargs['callback']
        self.stop = threading.Event()
    def __enter__(self):
        def feed():
            start = time.monotonic()
            for n, offset in enumerate(range(0, len(pcm), 640)):
                if self.stop.wait(max(0, start+n*.02-time.monotonic())):
                    return
                self.callback(pcm[offset:offset+640], 320, None, None)
        self.thread = threading.Thread(target=feed)
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def service(publish, chime, **kwargs):
    def state(value):
        states.append(value)
        publish(value)
    def ring(cancel):
        chimes.append(time.monotonic())
        chime(cancel)
    return TimerService(state, ring, **kwargs)


transcript = ROOT/'data/test-timer-voice.txt'
configure_plain_transcript(transcript)
with patch.object(client, 'microphone', return_value=(0, {'name': 'timer voice fixture'})), \
     patch.object(client.sd, 'RawInputStream', Input), \
     patch.object(client, 'TimerService', side_effect=service), \
     patch.object(client, 'SpeechOutput', side_effect=lambda cfg: SpeechOutput(cfg, sink=spoken.extend)), \
     patch.object(OpenHAB, 'command', side_effect=AssertionError('No device commands allowed')):
    client.listen(config, duration=len(pcm)/32000)
line = transcript.read_text(encoding='utf-8').splitlines()[-1]
assert 'Timer stopped.' in line, line
assert chimes and any('Ringing' in state for state in states), states
assert '00:00:03' in states or '00:00:02' in states, states
assert states[-1] == '00:00:00', states
result = api.session.get(api.url+'/rest/items/'+TIMER_ITEM, timeout=10)
result.raise_for_status()
assert result.json()['state'] == '00:00:00'
print('PASS: Alexa wake, timer start, openHAB countdown, chime and wakeless stop')
print('Display states:', states)
print(line)
