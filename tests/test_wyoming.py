"""Exercise real Wyoming TCP framing against an isolated fake R2T2 worker."""
import asyncio
from contextlib import asynccontextmanager
from functools import partial
import json

import pytest
from websockets.asyncio.server import serve
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice

from jarvis.wyoming import R2T2Handler

TOKEN = 'test-token-' * 4


@asynccontextmanager
async def service(*, fail=False, truncate_tts=False, **options):
    sessions = []

    async def backend(ws):
        assert ws.request.headers['Authorization'] == 'Bearer ' + TOKEN
        await ws.send(json.dumps({'type': 'ready', 'protocol': 2}))
        start = json.loads(await ws.recv())
        sessions.append(start['session'])
        if start['type'] == 'speak':
            assert ws.request.path == '/rpc'
            if fail:
                await ws.send(json.dumps({'type': 'error'}))
                return
            await ws.send(json.dumps({'type': 'tts_start', 'rate': 24000,
                                      'format': 'pcm_s16le', 'channels': 1}))
            await ws.send(b'\x01\x00' * 2400)
            await ws.send(json.dumps({'type': 'timing', 'message': 'test'}))
            await ws.send(b'\x02\x00' * 2400)
            if not truncate_tts:
                await ws.send(json.dumps({'type': 'tts_end'}))
            return
        audio = bytearray()
        async for raw in ws:
            if isinstance(raw, bytes):
                audio.extend(raw)
                await ws.send(json.dumps({'type': 'transcript', 'text': 'partial', 'final': False}))
                await ws.send(json.dumps({'type': 'timing', 'message': 'test'}))
            else:
                assert json.loads(raw)['type'] == 'asr_finish'
                result = {'type': 'error'} if fail else {
                    'type': 'transcript', 'text': f'{len(audio)}:{audio[:2].hex()}', 'final': True}
                await ws.send(json.dumps(result))

    async with serve(backend, '127.0.0.1', 0) as worker:
        port = worker.sockets[0].getsockname()[1]
        handlers = []

        def connection(reader, writer):
            handler = R2T2Handler(reader, writer, backend=f'ws://127.0.0.1:{port}/asr',
                                 token=TOKEN, **options)
            handlers.append(asyncio.create_task(handler.run()))

        server = await asyncio.start_server(connection, '127.0.0.1', 0)
        async with server:
            try:
                yield partial(AsyncTcpClient, '127.0.0.1', server.sockets[0].getsockname()[1]), sessions
            finally:
                for task in handlers:
                    task.cancel()
                await asyncio.gather(*handlers, return_exceptions=True)


async def transcribe(client, audio):
    await client.write_event(Transcribe(language='en').event())
    await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
    await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=audio).event())
    await client.write_event(AudioStop().event())
    return await asyncio.wait_for(client.read_event(), 5)


def test_describe_concurrent_and_reused_connections():
    async def run():
        async with service() as (client, sessions):
            async with client() as discovery:
                await discovery.write_event(Describe().event())
                info = Info.from_event(await discovery.read_event())
                assert info.asr[0].models[0].languages == ['en']
                assert info.tts[0].name == 'Kokoro'
                assert info.tts[0].voices[0].name == 'af_heart'
                assert not info.tts[0].supports_synthesize_streaming

            async def recognize(value):
                async with client() as connection:
                    for _ in range(2):
                        result = await transcribe(connection, bytes([value, 0]) * 16000)
                        assert Transcript.from_event(result).text == f'32000:{value:02x}00'

            await asyncio.gather(recognize(1), recognize(2))
            assert len(set(sessions)) == 4
    asyncio.run(run())


def test_tts_stream_and_reuse_with_asr():
    async def run():
        async with service() as (client, sessions):
            async with client() as connection:
                await connection.write_event(Synthesize('Hello', SynthesizeVoice(name='af_heart')).event())
                start = AudioStart.from_event(await connection.read_event())
                assert (start.rate, start.width, start.channels) == (24000, 2, 1)
                audio = bytearray()
                while True:
                    event = await connection.read_event()
                    if AudioStop.is_type(event.type):
                        break
                    audio.extend(AudioChunk.from_event(event).audio)
                assert audio == b'\x01\x00' * 2400 + b'\x02\x00' * 2400
                assert Transcript.from_event(await transcribe(connection, bytes(320))).text == '320:0000'
            assert len(set(sessions)) == 2
    asyncio.run(run())


@pytest.mark.parametrize('synthesis', [
    Synthesize(''), Synthesize('x' * 1001),
    Synthesize('Hello', SynthesizeVoice(name='missing')),
    Synthesize('Hello', SynthesizeVoice(language='de')),
    Synthesize('Hello', text_format='ssml'),
], ids=['empty', 'too-long', 'voice', 'language', 'ssml'])
def test_invalid_synthesis(synthesis):
    async def run():
        async with service() as (client, sessions):
            async with client() as connection:
                await connection.write_event(synthesis.event())
                result = await connection.read_event()
                assert result.data['code'] == 'invalid-request'
            assert not sessions
    asyncio.run(run())


@pytest.mark.parametrize('failure', [{'fail': True}, {'truncate_tts': True}])
def test_synthesis_backend_failures(failure):
    async def run():
        async with service(**failure) as (client, _):
            async with client() as connection:
                await connection.write_event(Synthesize('Hello').event())
                while (event := await connection.read_event()).type != 'error':
                    assert event.type in ('audio-start', 'audio-chunk')
                assert event.data['code'] == 'backend-error'
    asyncio.run(run())


@pytest.mark.parametrize('event', [
    Transcribe(language='de').event(),
    AudioStart(rate=48000, width=2, channels=1).event(),
    AudioChunk(rate=16000, width=2, channels=1, audio=b'\0\0').event(),
    AudioStop().event(),
])
def test_invalid_requests(event):
    async def run():
        async with service() as (client, _):
            async with client() as connection:
                await connection.write_event(event)
                error = await connection.read_event()
                assert error.type == 'error'
                assert error.data['code'] == 'invalid-request'
    asyncio.run(run())


@pytest.mark.parametrize('audio,max_seconds', [(b'\0', 33), (bytes(32002), 1)],
                         ids=['odd-sample', 'too-long'])
def test_invalid_pcm_and_duration_limit(audio, max_seconds):
    async def run():
        async with service(max_seconds=max_seconds) as (client, _):
            async with client() as connection:
                result = await transcribe(connection, audio)
                assert result.type == 'error'
                assert result.data['code'] == 'invalid-request'
    asyncio.run(run())


def test_backend_error_is_reported():
    async def run():
        async with service(fail=True) as (client, _):
            async with client() as connection:
                result = await transcribe(connection, bytes(320))
                assert result.type == 'error'
                assert result.data['code'] == 'backend-error'
    asyncio.run(run())


def test_idle_client_is_closed():
    async def run():
        async with service(timeout=0.05) as (client, _):
            async with client() as connection:
                result = await asyncio.wait_for(connection.read_event(), 2)
                assert result.data['code'] == 'stream-error'
                assert await connection.read_event() is None
    asyncio.run(run())
