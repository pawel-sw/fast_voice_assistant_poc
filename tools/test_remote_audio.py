"""Exercise client-owned control and VM inference; device commands are dry-run."""
import argparse
import json
import queue
from pathlib import Path
import sys
import threading
import time
import wave
from math import gcd
import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from jarvis.integrations import create_integration
from jarvis.control import Catalog
from jarvis.client_brain import ClientBrain
from jarvis.remote_rpc import RemotePlanner, SpeechOutput, connect_remote, ready, timing_event
from jarvis.observability import configure_plain_transcript

parser = argparse.ArgumentParser()
parser.add_argument('wav', nargs='?')
parser.add_argument('--text')
parser.add_argument('--tentative', action='store_true')
parser.add_argument('--play', action='store_true')
args = parser.parse_args()
load_dotenv(ROOT/'.env')
config = json.loads((ROOT/'config.json').read_text())
api = create_integration(config)
catalog = Catalog(api.items(), config)
pcm = bytearray()
speaker = SpeechOutput(config, sink=None if args.play else pcm.extend)
brain = ClientBrain(api, catalog, RemotePlanner(catalog, config), speaker, config, dry_run=True)
configure_plain_transcript(ROOT/'data/test-remote-live.txt')
started = time.monotonic()
try:
    session = brain.start(tentative=args.tentative)
    if args.text:
        brain.transcript({'session': session, 'utterance': 0, 'text': args.text, 'final': True})
    else:
        with wave.open(args.wav) as file:
            assert file.getnchannels() == 1 and file.getsampwidth() == 2
            audio = file.readframes(file.getnframes())
            rate = file.getframerate()
        if rate != 16000:
            divisor = gcd(rate, 16000)
            samples = resample_poly(np.frombuffer(audio, dtype='<i2').astype(np.float32), 16000//divisor, rate//divisor)
            audio = np.clip(samples, -32768, 32767).astype('<i2').tobytes()
        audio = bytes(16000) + audio + bytes(24000)
        with connect_remote(config) as ws:
            ready(ws)
            events = queue.Queue()
            def receive():
                try:
                    for data in ws:
                        events.put(json.loads(data))
                except Exception:
                    pass
            threading.Thread(target=receive, daemon=True).start()
            ws.send(json.dumps({'type': 'asr_start', 'session': session, 'utterance': 0}))
            for offset in range(0, len(audio), 15360):
                ws.send(audio[offset:offset+15360])
                time.sleep(.48)
                while not events.empty():
                    event = events.get_nowait()
                    if not timing_event(event):
                        brain.transcript(event)
                        print(event.get('text', ''), flush=True)
            ws.send(json.dumps({'type': 'asr_finish'}))
            while time.monotonic()-started < 45:
                event = events.get(timeout=20)
                if timing_event(event):
                    continue
                brain.transcript(event)
                if event.get('final'):
                    print('FINAL:', event['text'], flush=True)
                    break
    while time.monotonic()-started < 60:
        brain.tick()
        state = brain.current
        if not state.listening and not any((state.plan, state.delivery, state.speech)):
            break
        time.sleep(.02)
    if not args.play:
        if not pcm:
            raise SystemExit('No reply audio received')
        out = ROOT/'data/kokoro-reply.wav'
        with wave.open(str(out), 'wb') as f:
            f.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            f.writeframes(pcm)
        print(f'REPLY AUDIO: {len(pcm)/48000:.2f} seconds; {out}')
    print((ROOT/'data/test-remote-live.txt').read_text(encoding='utf-8').splitlines()[-1])
finally:
    brain.close()
