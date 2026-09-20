"""Drive the production wake/VAD/client loop with a WAV; no actuator mutations."""
from pathlib import Path
import json
from math import gcd
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
from jarvis.remote_rpc import SpeechOutput
from jarvis.observability import configure_plain_transcript

load_dotenv(ROOT/'.env')
config = json.loads((ROOT/'config.json').read_text())
with wave.open(sys.argv[1]) as file:
    rate = file.getframerate()
    samples = np.frombuffer(file.readframes(file.getnframes()), dtype='<i2').astype(np.float32)
if rate != 16000:
    d = gcd(rate, 16000)
    samples = resample_poly(samples, 16000//d, rate//d)
pcm = bytes(32000*3)+np.clip(samples, -32768, 32767).astype('<i2').tobytes()+bytes(32000*10)
pcm += bytes(-len(pcm) % 640)
reply = bytearray()


class Input:
    def __init__(self, **kwargs):
        self.callback = kwargs['callback']
        self.stop = threading.Event()
    def __enter__(self):
        def feed():
            started = time.monotonic()
            for n, offset in enumerate(range(0, len(pcm), 640)):
                if self.stop.wait(max(0, started+n*.02-time.monotonic())):
                    break
                self.callback(pcm[offset:offset+640], 320, None, None)
        self.thread = threading.Thread(target=feed)
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


path = ROOT/'data/test-client-listen.txt'
configure_plain_transcript(path)
with patch.object(client, 'microphone', return_value=(0, {'name': 'recorded C920 test input'})), \
     patch.object(client.sd, 'RawInputStream', Input), \
     patch.object(client, 'SpeechOutput', side_effect=lambda config: SpeechOutput(config, sink=reply.extend)):
    client.listen(config, duration=len(pcm)/32000, dry_run=True)
if not reply:
    raise SystemExit('FAIL: production client did not return speech')
print(f'PASS: production wake/VAD/control loop received {len(reply)/48000:.2f}s of speech')
print(path.read_text(encoding='utf-8').splitlines()[-1])
