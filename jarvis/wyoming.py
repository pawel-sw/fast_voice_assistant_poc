"""Wyoming STT/TTS endpoint backed by the shared R2T2 and Kokoro worker."""
import argparse
import asyncio
from contextlib import suppress
from functools import partial
import json
import logging
import os
from pathlib import Path
import uuid
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from websockets.asyncio.client import connect
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import async_read_event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

LOG = logging.getLogger(__name__)
ATTRIBUTION = Attribution('NetEase Youdao', 'https://github.com/netease-youdao/Confucius4-R2T2')
INFO = Info(asr=[AsrProgram(
    name='R2T2', description='Shared GPU streaming speech recognition',
    attribution=ATTRIBUTION, installed=True, version='1.0.0',
    models=[AsrModel(name='R2T2', description='English streaming recognition',
                     attribution=ATTRIBUTION, installed=True, version=None,
                     languages=['en'])],
)])


def service_info(voice):
    attribution = Attribution('hexgrad', 'https://huggingface.co/hexgrad/Kokoro-82M')
    return Info(asr=INFO.asr, tts=[TtsProgram(
        name='Kokoro', description='Shared GPU speech synthesis',
        attribution=attribution, installed=True, version='0.9.4',
        voices=[TtsVoice(name=voice, description=voice, attribution=attribution,
                         installed=True, version=None, languages=['en'])],
    )])


class R2T2Handler(AsyncEventHandler):
    """One isolated backend session per utterance; no second GPU model."""

    def __init__(self, reader, writer, *, backend, token, max_seconds=33, timeout=60,
                 voice='af_heart'):
        super().__init__(reader, writer)
        self.backend, self.token = backend, token
        self.max_bytes = int(max_seconds * 32000)
        self.timeout = timeout
        self.ws = None
        self.result = None
        self.started = False
        self.received = 0
        self.voice = voice
        self.info = service_info(voice)
        self.rpc_backend = urlunsplit(urlsplit(backend)._replace(path='/rpc'))

    async def run(self):
        try:
            while True:
                event = await asyncio.wait_for(async_read_event(self.reader), self.timeout)
                if event is None or not await self.handle_event(event):
                    break
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception:
            LOG.warning('Wyoming connection failed', exc_info=True)
            with suppress(Exception):
                await self.write_event(Error('Invalid or timed out Wyoming stream', 'stream-error').event())
        finally:
            await self.disconnect()
            self.writer.close()
            with suppress(ConnectionError):
                await self.writer.wait_closed()

    async def disconnect(self):
        if self.result is not None:
            self.result.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self.result
            self.result = None
        if self.ws is not None:
            await self.ws.close()
            self.ws = None
        self.started = False
        self.received = 0

    async def receive_result(self):
        # Drain partial transcripts and timings while audio is still arriving.
        # Otherwise the worker's output queue can block a long utterance.
        async for raw in self.ws:
            data = json.loads(raw)
            if data['type'] == 'error':
                raise RuntimeError('R2T2 rejected the transcription request')
            if data['type'] == 'transcript' and data.get('final'):
                return data['text']
        raise RuntimeError('R2T2 disconnected before the final transcript')

    @staticmethod
    def validate_format(audio):
        if (audio.rate, audio.width, audio.channels) != (16000, 2, 1):
            raise ValueError('Audio must be 16000 Hz, 16-bit little-endian PCM, mono')

    async def synthesize(self, request):
        if not isinstance(request.text, str) or not request.text.strip() or len(request.text) > 1000:
            raise ValueError('Speech text must contain 1 to 1000 characters')
        if request.text_format not in (None, 'text'):
            raise ValueError('Only plain text synthesis is supported')
        if request.voice:
            if request.voice.name not in (None, self.voice) or request.voice.speaker is not None:
                raise ValueError('Unknown voice or speaker')
            if request.voice.language and request.voice.language.split('-')[0] != 'en':
                raise ValueError('Only English is configured')
        await self.disconnect()
        async with asyncio.timeout(self.timeout):
            self.ws = await connect(self.rpc_backend,
                additional_headers={'Authorization': 'Bearer ' + self.token},
                open_timeout=10, close_timeout=2, max_size=128000, compression=None)
            ready = json.loads(await self.ws.recv())
            if ready.get('type') != 'ready' or ready.get('protocol') != 2:
                raise RuntimeError('Kokoro backend is not ready')
            await self.ws.send(json.dumps({'type': 'speak', 'session': uuid.uuid4().hex,
                                          'text': request.text.strip()}))
            audio_started = False
            audio_bytes = 0
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    if not audio_started or len(raw) % 2:
                        raise RuntimeError('Invalid Kokoro audio stream')
                    audio_bytes += len(raw)
                    await self.write_event(AudioChunk(rate=24000, width=2, channels=1, audio=raw).event())
                    continue
                data = json.loads(raw)
                if data['type'] == 'tts_start':
                    if audio_started or (data.get('rate'), data.get('format'), data.get('channels')) != (24000, 'pcm_s16le', 1):
                        raise RuntimeError('Unsupported Kokoro audio format')
                    audio_started = True
                    await self.write_event(AudioStart(rate=24000, width=2, channels=1).event())
                elif data['type'] == 'tts_end':
                    if not audio_bytes:
                        raise RuntimeError('Kokoro returned no audio')
                    await self.write_event(AudioStop().event())
                    break
                elif data['type'] == 'error':
                    raise RuntimeError('Kokoro rejected the synthesis request')
            else:
                raise RuntimeError('Kokoro disconnected before audio finished')
        await self.disconnect()

    async def handle_event(self, event):
        try:
            if Describe.is_type(event.type):
                await self.write_event(self.info.event())
            elif Synthesize.is_type(event.type):
                await self.synthesize(Synthesize.from_event(event))
            elif Transcribe.is_type(event.type):
                request = Transcribe.from_event(event)
                if request.language and request.language.split('-')[0] != 'en':
                    raise ValueError('Only English is configured')
                if request.name not in (None, 'R2T2'):
                    raise ValueError('Unknown ASR model')
                await self.disconnect()
            elif AudioStart.is_type(event.type):
                self.validate_format(AudioStart.from_event(event))
                if self.started:
                    raise ValueError('Audio stream already started')
                self.ws = await connect(self.backend,
                    additional_headers={'Authorization': 'Bearer ' + self.token},
                    open_timeout=10, close_timeout=2, max_size=128000, compression=None)
                ready = json.loads(await asyncio.wait_for(self.ws.recv(), 10))
                if ready.get('type') != 'ready' or ready.get('protocol') != 2:
                    raise RuntimeError('R2T2 backend is not ready')
                await self.ws.send(json.dumps({'type': 'asr_start',
                    'session': uuid.uuid4().hex, 'utterance': 0}))
                self.result = asyncio.create_task(self.receive_result())
                self.started = True
            elif AudioChunk.is_type(event.type):
                if not self.started:
                    raise ValueError('AudioStart is required before audio')
                chunk = AudioChunk.from_event(event)
                self.validate_format(chunk)
                if len(chunk.audio) % 2:
                    raise ValueError('Incomplete PCM16 sample')
                self.received += len(chunk.audio)
                if self.received > self.max_bytes:
                    raise ValueError('Utterance too long')
                if self.result.done():
                    self.result.result()
                    raise RuntimeError('R2T2 ended before AudioStop')
                for offset in range(0, len(chunk.audio), 5120):
                    await asyncio.wait_for(self.ws.send(chunk.audio[offset:offset+5120]), self.timeout)
            elif AudioStop.is_type(event.type):
                if not self.started:
                    raise ValueError('AudioStart is required before AudioStop')
                await self.ws.send(json.dumps({'type': 'asr_finish'}))
                text = await asyncio.wait_for(self.result, self.timeout)
                await self.write_event(Transcript(text=text, language='en').event())
                await self.disconnect()
            # Wyoming extensions unknown to this adapter are ignored.
            return True
        except (ValueError, KeyError, TypeError) as exc:
            await self.write_event(Error(str(exc), 'invalid-request').event())
        except Exception:
            LOG.warning('Inference backend request failed', exc_info=True)
            await self.write_event(Error('Inference backend unavailable or timed out', 'backend-error').event())
        await self.disconnect()
        return False


def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uri', default='tcp://0.0.0.0:10300')
    parser.add_argument('--backend', default='ws://127.0.0.1:8765/asr')
    parser.add_argument('--max-seconds', type=float, default=33)
    args = parser.parse_args()
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    token = os.environ['JARVIS_REMOTE_TOKEN']
    if len(token) < 32 or args.max_seconds <= 0:
        parser.error('A 32-character backend token and positive max-seconds are required')
    logging.basicConfig(level=logging.INFO)
    server = AsyncServer.from_uri(args.uri)
    LOG.info('Wyoming R2T2/Kokoro listening on %s', args.uri)
    asyncio.run(server.run(partial(R2T2Handler, backend=args.backend,
                                  token=token, max_seconds=args.max_seconds,
                                  voice=config.get('tts_voice', 'af_heart'))))


if __name__ == '__main__':
    main()
