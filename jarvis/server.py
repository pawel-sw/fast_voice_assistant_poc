"""Authenticated inference service: R2T2, Needle and Kokoro; no home control."""
import hmac
import json
import logging
import math
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading
from contextlib import contextmanager
from http import HTTPStatus

import numpy as np
from dotenv import load_dotenv
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from .scheduling import FairGate
from .inference import GENERIC_SYSTEM

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


@contextmanager
def inference_deadline(seconds=20):
    # A stuck CUDA worker cannot be cancelled safely in-process. systemd restarts
    # the service and its children; clients discard the interrupted request.
    def stalled():
        log.critical('Inference exceeded %s seconds; restarting worker', seconds)
        os._exit(1)
    timer = threading.Timer(seconds, stalled)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def main():
    load_dotenv(ROOT / '.env')
    token = os.environ['JARVIS_REMOTE_TOKEN']
    if len(token) < 32:
        raise ValueError('JARVIS_REMOTE_TOKEN must be at least 32 characters')
    (ROOT / 'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[
        logging.StreamHandler(), RotatingFileHandler(ROOT/'logs/jarvis.log', maxBytes=2_000_000, backupCount=3)])
    timing = logging.getLogger('jarvis.timing')
    timing.propagate = False
    timing.setLevel(logging.INFO)
    timing.addHandler(RotatingFileHandler(ROOT/'logs/timings.log', maxBytes=5_000_000, backupCount=3))
    config = json.loads((ROOT/'config.json').read_text())
    from .remote_asr import RemoteASR
    from .inference import NeedleEngine
    from .tts import Kokoro
    planner = NeedleEngine()
    asr = RemoteASR(config)
    tts = Kokoro(config)
    gpu_lock, needle_lock = FairGate('gpu'), FairGate('needle')
    max_connections = int(config.get('server_max_connections', 16))
    connections = threading.BoundedSemaphore(max_connections)
    capabilities = {'protocol': 2, 'asr': {'path': '/asr', 'model': 'R2T2',
        'rate': 16000, 'format': 'pcm_s16le', 'channels': 1,
        'max_seconds': config['max_utterance_seconds'] + 3},
        'plan': {'path': '/rpc', 'model': 'Needle3-20L-121M', 'max_tools': 100},
        'tts': {'path': '/rpc', 'model': 'Kokoro-82M', 'rate': tts.rate,
            'format': 'pcm_s16le', 'channels': 1, 'voice': tts.voice, 'speed': tts.speed},
        'max_connections': max_connections, 'queue_timeout_seconds': 30}

    def authorize(connection, request):
        if not hmac.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + token):
            return connection.respond(HTTPStatus.UNAUTHORIZED, 'Unauthorized\n')
        if request.path == '/health':
            return connection.respond(HTTPStatus.OK, json.dumps({'ready': True, 'protocol': 2,
                'backend': 'vllm', 'tts': 'Kokoro-82M', 'tts_device': tts.device, 'project': str(ROOT)}))
        if request.path == '/services':
            response = connection.respond(HTTPStatus.OK, json.dumps(capabilities))
            response.headers['Content-Type'] = 'application/json'
            return response
        if request.path not in ('/', '/asr', '/rpc'):
            return connection.respond(HTTPStatus.NOT_FOUND, 'Unknown endpoint\n')

    def handler(ws):
        is_asr = ws.request.path != '/rpc'
        if not connections.acquire(blocking=False):
            ws.close(1013, 'Service at connection capacity; try again later')
            return
        session_id = None
        utterance = 0
        active = False
        state = None
        samples_seen = 0
        owner_thread = threading.get_ident()

        def emit(event):
            ws.send(json.dumps({'session': session_id, **event}, separators=(',', ':')))

        class TimingForwarder(logging.Handler):
            def filter(self, record):
                # Filter before logging acquires this handler's lock: an unrelated
                # client's network write must not stall this request's timings.
                return record.thread == owner_thread

            def emit(self, record):
                try:
                    emit({'type': 'timing', 'message': record.getMessage()})
                except ConnectionClosed:
                    pass

        forwarder = TimingForwarder()
        timing.addHandler(forwarder)
        try:
            emit({'type': 'ready', 'protocol': 2, 'backend': 'vllm', 'tts': 'Kokoro-82M'})
            while True:
                try:
                    message = ws.recv(timeout=5)
                except TimeoutError:
                    if active:
                        active = False
                        state = None
                        emit({'type': 'error', 'message': 'Audio stream timed out'})
                    continue
                if isinstance(message, bytes):
                    if not is_asr or not active:
                        continue
                    if len(message) % 2:
                        raise ValueError('Invalid PCM16 audio')
                    samples_seen += len(message)//2
                    if samples_seen > 16000*(config['max_utterance_seconds']+3):
                        raise ValueError('Utterance too long')
                    feed_bytes = max(2, round(16000 * config['chunk_seconds']) * 2)
                    for offset in range(0, len(message), feed_bytes):
                        samples = np.frombuffer(message[offset:offset+feed_bytes], dtype='<i2').astype(np.float32)/32768
                        with gpu_lock, inference_deadline():
                            text = asr.feed(samples, state)
                        emit({'type': 'transcript', 'utterance': utterance, 'text': text, 'final': False})
                    continue
                data = json.loads(message)
                kind = data['type']
                if is_asr and kind == 'asr_start':
                    session_id, utterance = str(data['session']), int(data.get('utterance', 0))
                    with gpu_lock:
                        state = asr.new_state()
                    active, samples_seen = True, 0
                elif is_asr and kind == 'asr_finish' and active:
                    with gpu_lock, inference_deadline():
                        text = asr.feed(np.zeros(0, dtype=np.float32), state, final=True)
                    active = False
                    state = None
                    emit({'type': 'transcript', 'utterance': utterance, 'text': text, 'final': True})
                    log.info('Heard: %s', text)
                elif is_asr and kind == 'cancel':
                    active = False
                    state = None
                elif not is_asr and kind == 'plan':
                    session_id = str(data['session'])
                    text, schemas = str(data['text']), data['schemas']
                    if len(text) > 4000 or not isinstance(schemas, list) or not 1 <= len(schemas) <= 100:
                        raise ValueError('Invalid inference request')
                    system = data.get('system', GENERIC_SYSTEM)
                    if not isinstance(system, str) or len(system) > 4000:
                        raise ValueError('Invalid system prompt')
                    with needle_lock, inference_deadline():
                        response = planner.infer(text, schemas, system=system)
                    emit({'type': 'plan', 'response': response})
                    return
                elif not is_asr and kind == 'speak':
                    session_id = str(data['session'])
                    text = str(data['text']).strip()
                    if not text or len(text) > 1000:
                        raise ValueError('Invalid speech text')
                    speed = data.get('speed', tts.speed)
                    if type(speed) not in (int, float) or not math.isfinite(speed) or not 0.5 <= speed <= 2:
                        raise ValueError('Speech speed must be between 0.5 and 2.0')
                    emit({'type': 'tts_start', 'rate': tts.rate, 'format': 'pcm_s16le', 'channels': 1, 'speed': speed})
                    chunks = iter(tts.chunks(text, speed=speed))
                    while True:
                        with gpu_lock, inference_deadline(30):
                            try:
                                pcm = next(chunks)
                            except StopIteration:
                                break
                        ws.send(pcm)
                    emit({'type': 'tts_end'})
                    return
                else:
                    raise ValueError('Unsupported protocol message')
        except ConnectionClosed:
            log.info('Client disconnected; pending inference discarded')
        except (TimeoutError, ValueError, KeyError, TypeError) as exc:
            try:
                emit({'type': 'error', 'code': 'busy' if isinstance(exc, TimeoutError) else 'invalid_request',
                      'message': str(exc)})
            except ConnectionClosed:
                pass
        except Exception:
            log.exception('Inference request failed')
            try:
                emit({'type': 'error', 'message': 'Inference failed; please try again'})
            except ConnectionClosed:
                pass
        finally:
            timing.removeHandler(forwarder)
            state = None
            connections.release()

    with serve(handler, os.getenv('JARVIS_BIND', '0.0.0.0'), int(os.getenv('JARVIS_PORT', '8765')),
               process_request=authorize, max_size=128_000, max_queue=128, compression=None,
               close_timeout=2, ping_interval=10, ping_timeout=20) as server:
        log.info('READY: R2T2/Needle/Kokoro inference service, protocol 2')
        server.serve_forever()


if __name__ == '__main__':
    main()
