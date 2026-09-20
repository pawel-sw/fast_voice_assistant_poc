"""Portable Python 3.10+ client. Only dependency: websockets==15.0.1."""
import json
import os
import queue
import threading
import uuid
from urllib.parse import urlsplit, urlunsplit
from websockets.sync.client import connect


def speech_messages(ws, timeout=60, cancelled=None, max_bytes=16_000_000):
    """Drain the socket independently of a slow speaker, with a bounded reply.

    Playback-paced recv can fill the WebSocket queue and delay its close
    handshake beyond the server's timeout, losing the tail of long replies.
    """
    inbox = queue.SimpleQueue()
    stopped = threading.Event()
    cancelled = cancelled if cancelled is not None else threading.Event()

    def receive():
        size = 0
        try:
            while not stopped.is_set() and not cancelled.is_set():
                message = ws.recv(timeout=timeout)
                size += len(message) if isinstance(message, bytes) else len(message.encode('utf-8'))
                if size > max_bytes:
                    raise ValueError('Speech response exceeded buffer limit')
                event = None if isinstance(message, bytes) else json.loads(message)
                inbox.put(message)
                if event and event.get('type') in ('tts_end', 'error'):
                    return
        except Exception as exc:
            inbox.put(exc)

    receiver = threading.Thread(target=receive, name='tts-receiver', daemon=True)
    receiver.start()
    try:
        while not cancelled.is_set():
            try:
                message = inbox.get(timeout=.1)
            except queue.Empty:
                continue
            if isinstance(message, Exception):
                raise message
            yield message
            if isinstance(message, str) and json.loads(message).get('type') in ('tts_end', 'error'):
                return
    finally:
        stopped.set()
        ws.close()
        receiver.join(timeout=2)


class InferenceClient:
    """Thread-safe: every operation owns a socket and a unique session ID."""
    def __init__(self, url=None, token=None, timeout=60):
        self.url = url or os.environ['INFERENCE_URL']
        self.token = token or os.environ['JARVIS_REMOTE_TOKEN']
        self.timeout = timeout

    def _connect(self, path):
        url = urlsplit(self.url)
        ws = connect(urlunsplit((url.scheme, url.netloc, path, '', '')),
            additional_headers={'Authorization': 'Bearer ' + self.token},
            compression=None, max_size=128_000, open_timeout=10, close_timeout=2,
            ping_interval=10, ping_timeout=20, proxy=None)
        try:
            event = self._event(ws.recv(timeout=self.timeout))
            if event.get('type') != 'ready' or event.get('protocol') != 2:
                raise RuntimeError('Service requires protocol 2')
            return ws
        except BaseException:
            ws.close()
            raise

    @staticmethod
    def _event(message):
        event = json.loads(message)
        if event.get('type') == 'error':
            raise RuntimeError(f"{event.get('code', 'inference_error')}: {event['message']}")
        return event

    def plan(self, text, schemas, system=None):
        request = {'type': 'plan', 'session': uuid.uuid4().hex, 'text': text, 'schemas': schemas}
        if system is not None:
            request['system'] = system
        with self._connect('/rpc') as ws:
            ws.send(json.dumps(request))
            while True:
                event = self._event(ws.recv(timeout=self.timeout))
                if event['type'] == 'plan':
                    return event['response']

    def speak(self, text, speed=None):
        """Yield mono PCM16 little-endian chunks at 24 kHz; default speed 1.2."""
        request = {'type': 'speak', 'session': uuid.uuid4().hex, 'text': text}
        if speed is not None:
            request['speed'] = speed
        with self._connect('/rpc') as ws:
            ws.send(json.dumps(request))
            messages = speech_messages(ws, timeout=self.timeout)
            try:
                for message in messages:
                    if isinstance(message, bytes):
                        yield message
                    else:
                        event = self._event(message)
                        if event['type'] == 'tts_end':
                            return
            finally:
                messages.close()

    def transcribe(self, audio_chunks):
        """Yield transcript events while consuming mono PCM16/16 kHz chunks.

        The iterable may block awaiting microphone data. Keep chunks <=15360
        bytes and gaps below five seconds. Closing the generator closes its
        connection. Producers should be finite or independently cancellable.
        """
        with self._connect('/asr') as ws:
            ws.send(json.dumps({'type': 'asr_start', 'session': uuid.uuid4().hex, 'utterance': 0}))
            stopped = threading.Event()
            errors = []

            def send():
                try:
                    for chunk in audio_chunks:
                        if stopped.is_set():
                            return
                        if len(chunk) % 2:
                            raise ValueError('PCM16 chunks must contain complete samples')
                        for offset in range(0, len(chunk), 15360):
                            ws.send(chunk[offset:offset + 15360])
                    ws.send(json.dumps({'type': 'asr_finish'}))
                except Exception as exc:
                    errors.append(exc)
                    ws.close()

            sender = threading.Thread(target=send, daemon=True)
            sender.start()
            try:
                while True:
                    event = self._event(ws.recv(timeout=self.timeout))
                    if event['type'] == 'transcript':
                        yield event
                        if event['final']:
                            return
            except Exception:
                if errors:
                    raise errors[0]
                raise
            finally:
                stopped.set()
                ws.close()
                sender.join(timeout=1)
