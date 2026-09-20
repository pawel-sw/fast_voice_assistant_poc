# Shared VM inference API

Base URL: `ws://inference.example.invalid:8765`. Python 3.10+ clients need only
`pip install websockets==15.0.1` and a copy of `clients/inference_client.py`.
The same files are available on the VM under `/p/home_assistant`.
No assistant, openHAB, microphone, CUDA or model packages are needed in other projects.

Set `JARVIS_REMOTE_TOKEN` in each project's environment to the existing secret
from this project's ignored `.env` (or `/p/home_assistant/.env` on the VM).
Do not commit that value. All HTTP/WebSocket requests require
`Authorization: Bearer <token>`. This is a shared LAN credential, not project isolation.
The service listens on all VM interfaces, port 8765. Use TLS before exposing it beyond the trusted LAN.

## Python examples

```python
import wave
from inference_client import InferenceClient

client = InferenceClient('ws://inference.example.invalid:8765')

# Streaming synthesis; default speaking rate is 1.2x.
with wave.open('reply.wav', 'wb') as out:
    out.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
    for pcm in client.speak('Your export is ready.'):
        out.writeframesraw(pcm)  # Or send each chunk directly to a speaker.

# Streaming recognition; WAV must already be mono, PCM16, 16 kHz.
with wave.open('request.wav', 'rb') as source:
    assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
    for result in client.transcribe(iter(lambda: source.readframes(7680), b'')):
        print(result['text'], 'FINAL' if result['final'] else '')

# General tool selection: you supply schemas and validate/execute the result.
result = client.plan('Search for blue shoes.', [{
    'name': 'search_catalog', 'description': 'Search the product catalog',
    'parameters': {'type': 'object', 'properties': {
        'query': {'type': 'string'}}, 'required': ['query']}
}], system='Select the appropriate catalog tool for the request.')
print(result)
```

The client can be shared between threads. Each call owns its connection and
session. Synthesis reception runs separately from playback: slow consumers can
play a complete reply after the server finishes and closes its socket. A reply
buffer limit of 16 MB prevents unlimited accumulation. Consume synthesis/transcription generators fully, or call `.close()`
when abandoning them. A live audio iterable must itself support cancellation
if it can block indefinitely. No requests are automatically replayed.

## Wire protocol (version 2)

Authenticated `GET http://inference.example.invalid:8765/health` reports readiness after model
warmup. `GET /services` returns models, formats, limits and default TTS speed.
Every accepted WebSocket begins with `{"type":"ready","protocol":2,...}`.
JSON events may include `session`; `timing` events can be ignored or logged.
Errors have `type:error`, `message`, and optionally `code` (`busy` or `invalid_request`).

| Connection | Client messages | Responses |
| --- | --- | --- |
| `/asr` | `{"type":"asr_start","session":"unique-id","utterance":0}`, binary PCM16 mono 16 kHz frames, then `{"type":"asr_finish"}` | `transcript` events with `text`, `utterance`, `final` |
| `/asr` | `{"type":"cancel"}` | Discards this connection's utterance; no acknowledgement |
| `/rpc` | `{"type":"plan","session":"unique-id","text":"...","schemas":[...],"system":"optional prompt"}` | `{"type":"plan","response":{...}}`, then closes |
| `/rpc` | `{"type":"speak","session":"unique-id","text":"...","speed":1.2}` | `tts_start` (rate 24000, channels 1, format pcm_s16le, speed), binary audio chunks, `tts_end`, then closes |

ASR sockets may handle successive utterances. Use a separate socket per
simultaneous stream. Send audio in chunks up to 15360 bytes; consume responses
concurrently with sending. Text is cumulative and may revise previous words.
Audio gaps over five seconds discard the utterance. Maximum audio: 28 seconds.
Maximum message size: 128000 bytes. Plan text/system: 4000 characters each,
1–100 tool schemas. Speech: 1–1000 characters, speed 0.5–2.0 (default 1.2),
voice `af_heart`. The VM returns model output; it never executes tools.

## Concurrency and capacity

Up to 16 accepted connections share the loaded models; extra connections close
with WebSocket code 1013. Each ASR stream has independent history and buffers.
GPU operations use a FIFO queue shared by ASR and TTS, released between audio
chunks; network writes do not hold that queue. Needle has a separate FIFO queue
and can run alongside GPU work. Queue waiting is capped at 30 seconds, with
`queue.wait` timings logged. This supports concurrent clients by interleaving
inference, not parallel GPU kernels or duplicated models. Latency grows with load.
Tune `server_max_connections` in config.json if needed; 16 is a connection cap,
not a promise of 16 real-time audio streams on this GPU.

Running model operations have a 20-second watchdog (30 for TTS). A stuck runtime
restarts the entire service and interrupts all clients. Callers should reconnect
and decide whether retrying is appropriate. Use a receive timeout of at least
60 seconds when sharing a busy VM. Jarvis retains shorter interactive timeouts.

## Operations

```bash
sudo systemctl status jarvis-vm
sudo systemctl restart jarvis-vm
journalctl -u jarvis-vm -f
```

Code and models remain in `/p/home_assistant`; the systemd service starts on boot.
Inference logs are in its `logs/` directory. Client-side tool execution, assistant
behavior and credentials stay in the calling project.
