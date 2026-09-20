"""Small client RPCs and streamed speaker playback; no model runtime imports."""
import json
import logging
import math
import os
import re
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import sounddevice as sd
from websockets.sync.client import connect
from .observability import timed, planning_feedback
from .inference import SYSTEM
from clients.inference_client import speech_messages

log = logging.getLogger(__name__)


def connect_remote(config, path='/asr'):
    url = urlsplit(config['remote_url'])
    return connect(urlunsplit((url.scheme, url.netloc, path, '', '')),
        additional_headers={'Authorization': 'Bearer '+os.environ['JARVIS_REMOTE_TOKEN']},
        compression=None, max_size=128_000, open_timeout=5, close_timeout=2,
        ping_interval=10, ping_timeout=20, proxy=None)


def ready(ws):
    event = json.loads(ws.recv(timeout=10))
    if event.get('type') != 'ready' or event.get('protocol') != 2:
        raise RuntimeError(event.get('message', 'VM requires protocol 2'))


def timing_event(event):
    if event.get('type') == 'timing':
        logging.getLogger('jarvis.timing').info('REMOTE %s', event['message'])
        return True
    if event.get('type') == 'error':
        raise RuntimeError(event['message'])
    return False


def validate_response(catalog, text, schemas, response, minimum):
    if not isinstance(response, dict):
        return []
    confidence = response.get('confidence')
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not minimum <= confidence <= 1:
        return []
    calls = response.get('function_calls')
    if not isinstance(calls, list) or any(not isinstance(c, dict) or c.get('name') not in {s['name'] for s in schemas} for c in calls):
        return []
    try:
        actions = catalog.validate(calls)
        for call in calls:
            if call['name'].startswith('switch_') and not re.search(r'\b'+re.escape(call['arguments']['value'])+r'\b', text, re.I):
                return []
        return actions
    except (ValueError, TypeError, KeyError):
        return []


class RemotePlanner:
    def __init__(self, catalog, config):
        self.catalog, self.config = catalog, config
        from .groq_fallback import GroqFallback
        self.groq = GroqFallback(config, catalog)

    def plan(self, text, session):
        text = ' '.join(text.split())
        if not text:
            return []
        if self.catalog.policy.ignored_request(text):
            log.info('Ignored room request; no inference or commands')
            return []
        text = self.catalog.policy.normalize(text)
        from .clock_queries import parse_clock
        clock = parse_clock(text)
        if clock is not None:
            return [clock]
        from .weather import parse_weather
        weather = parse_weather(text, self.config)
        if weather is not None:
            return [weather]
        from .timers import parse_timer
        timer = parse_timer(text)
        if timer is not None:
            return [timer]
        actions = self._try_needle(text, session)
        if actions or not self.groq.token:
            return actions
        log.info('Needle returned no valid call; trying Groq phonetic repair')
        corrected = text
        planning_feedback('groq repair', 'pending')
        try:
            corrected = self.groq.repair(text)
            log.info('Groq repaired transcript: %s', corrected)
            planning_feedback('groq repair', corrected)
        except Exception as exc:
            log.warning('Groq phonetic repair failed (%s)', type(exc).__name__)
            planning_feedback('groq repair', f'failed ({type(exc).__name__})')
        if self.catalog.policy.ignored_request(corrected):
            return []
        corrected = self.catalog.policy.normalize(corrected)
        clock = parse_clock(corrected)
        if clock is not None:
            return [clock]
        weather = parse_weather(corrected, self.config)
        if weather is not None:
            return [weather]
        timer = parse_timer(corrected)
        if timer is not None:
            return [timer]
        actions = self._try_needle(corrected, session)
        if actions:
            return actions
        log.info('Needle retry returned no valid call; trying Groq tool proposal')
        planning_feedback('groq call', 'pending')
        try:
            response = self.groq.propose(text, corrected)
            # Keep the same named-target, argument, confidence and ON/OFF checks.
            schemas = self.catalog.candidates(corrected)
            with timed('groq.tools.validate'):
                actions = validate_response(self.catalog, corrected, schemas, response, self.config['min_confidence'])
            log.info('Groq tool proposal %s', 'accepted' if actions else 'rejected')
            calls = response.get('function_calls', []) if isinstance(response, dict) else []
            planning_feedback('groq call', ('' if actions else 'rejected ') +
                              json.dumps(calls, separators=(',', ':'), ensure_ascii=False))
            return actions
        except Exception as exc:
            log.warning('Groq tool proposal failed (%s)', type(exc).__name__)
            planning_feedback('groq call', f'failed ({type(exc).__name__})')
            return []

    def _try_needle(self, text, session):
        try:
            return self._needle(text, session)
        except Exception as exc:
            log.warning('Needle request failed (%s)', type(exc).__name__)
            return []

    def _needle(self, text, session):
        text = ' '.join(text.split())
        if not text:
            return []
        if text[-1] not in '.!?':
            text += '.'
        with timed('catalog.select'):
            schemas = self.catalog.candidates(text)
        if not schemas:
            return []
        with timed('remote.needle.roundtrip'), connect_remote(self.config, '/rpc') as ws:
            ready(ws)
            ws.send(json.dumps({'type': 'plan', 'session': session, 'text': text, 'schemas': schemas, 'system': SYSTEM}))
            while True:
                event = json.loads(ws.recv(timeout=15))
                if timing_event(event):
                    continue
                if event['type'] == 'plan':
                    with timed('tools.validate'):
                        return validate_response(self.catalog, text, schemas, event['response'], self.config['min_confidence'])


class SpeechOutput:
    def __init__(self, config, sink=None):
        self.config, self.sink = config, sink
        self.busy = threading.Event()
        self.lock = threading.Lock()
        self.quiet_until = 0.0

    @property
    def muted(self):
        return self.busy.is_set() or time.monotonic() < self.quiet_until

    def speak(self, text, session, cancelled):
        with self.lock:
            return self._speak(text, session, cancelled)

    def chime(self, cancelled):
        """Soft ascending major chord; short pulses leave room to say stop."""
        import numpy as np
        rate = 24000
        samples = np.zeros(round(rate*1.8), dtype=np.float32)
        t = np.arange(round(rate*1.1), dtype=np.float32)/rate
        envelope = np.minimum(t/.03, 1) * np.exp(-4*t) * np.minimum((1.1-t)/.06, 1)
        for offset, frequency in ((0, 523.25), (.3, 659.25), (.6, 783.99)):
            start = round(offset*rate)
            samples[start:start+len(t)] += envelope * (np.sin(2*np.pi*frequency*t) + .12*np.sin(4*np.pi*frequency*t))
        samples *= min(.4, max(.02, self.config.get('timer_chime_volume', .18)))
        pcm = (np.clip(samples, -1, 1)*32767).astype('<i2').tobytes()
        with self.lock:
            if cancelled.is_set():
                return
            self.busy.set()
            stream = None
            try:
                if self.sink is None:
                    stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype='int16',
                                               device=self.config.get('speaker_device'), blocksize=1200)
                    stream.start()
                for offset in range(0, len(pcm), 2400):
                    if cancelled.is_set():
                        break
                    if self.sink is not None:
                        self.sink(pcm[offset:offset+2400])
                    else:
                        stream.write(pcm[offset:offset+2400])
                if stream:
                    stream.abort() if cancelled.is_set() else stream.stop()
            finally:
                if stream:
                    stream.close()
                self.quiet_until = time.monotonic()+.3
                self.busy.clear()

    def _speak(self, text, session, cancelled):
        if cancelled.is_set() or not self.config.get('tts_enabled', True):
            return
        self.busy.set()
        stream = None
        messages = None
        try:
            with timed('remote.kokoro_and_playback'), connect_remote(self.config, '/rpc') as ws:
                ready(ws)
                ws.send(json.dumps({'type': 'speak', 'session': session, 'text': text}))
                first = True
                audio_bytes = 0
                completed = False
                began = time.perf_counter()
                messages = speech_messages(ws, timeout=20, cancelled=cancelled)
                for message in messages:
                    if isinstance(message, bytes):
                        if first:
                            logging.getLogger('jarvis.timing').info('TIMING tts.first_audio %.2f ms', (time.perf_counter()-began)*1000)
                            first = False
                        if self.sink is not None:
                            self.sink(message)
                        elif stream is not None:
                            stream.write(message)
                        audio_bytes += len(message)
                        continue
                    event = json.loads(message)
                    if timing_event(event):
                        continue
                    if event['type'] == 'tts_start':
                        if event['rate'] != 24000 or event['format'] != 'pcm_s16le':
                            raise ValueError('Unsupported speech format')
                        if self.sink is None:
                            stream = sd.RawOutputStream(samplerate=24000, channels=1, dtype='int16',
                                device=self.config.get('speaker_device'), blocksize=2400)
                            stream.start()
                    elif event['type'] == 'tts_end':
                        completed = True
                        break
                if not completed and not cancelled.is_set():
                    raise RuntimeError('Speech stream ended without completion')
            if stream is not None:
                stream.abort() if cancelled.is_set() else stream.stop()
            logging.getLogger('jarvis.timing').info('TIMING tts.playback audio_seconds=%.2f completed=%s', audio_bytes/48000, completed and not cancelled.is_set())
            if cancelled.is_set():
                log.info('Speech cancelled after %.2f seconds of audio', audio_bytes/48000)
            else:
                log.info('Spoken: %s', text)
        finally:
            if messages is not None:
                messages.close()
            if stream is not None:
                stream.close()
            self.quiet_until = time.monotonic()+.5
            self.busy.clear()
