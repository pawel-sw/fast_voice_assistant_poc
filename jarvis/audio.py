import logging
import queue
import time
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad

from .wake import WakeWord
from .observability import timed, plain_cancel_candidate
from .input_audio import InputGain

log = logging.getLogger(__name__)
RATE = 16000
FRAME = 320  # WebRTC VAD accepts 20 ms PCM frames.


def microphone(name):
    devices = sd.query_devices()
    matches = [(i, d) for i, d in enumerate(devices)
               if name.lower() in d['name'].lower() and d['max_input_channels'] > 0]
    if not matches:
        raise RuntimeError(f'Microphone {name!r} not found. Run --devices.')
    # MME performs reliable 16 kHz resampling for the C920 on Windows.
    matches.sort(key=lambda entry: sd.query_hostapis(entry[1]['hostapi'])['name'] != 'MME')
    for index, device in matches:
        try:
            sd.check_input_settings(device=index, channels=1, samplerate=RATE, dtype='int16')
            return index, device
        except sd.PortAudioError:
            pass
    raise RuntimeError('C920 is present but cannot capture mono 16 kHz audio')


def listen(asr, conversation, config, duration=None, stop_event=None):
    index, device = microphone(config['microphone'])
    frames = queue.Queue(maxsize=500)
    overflow = False

    def callback(data, count, timing, status):
        nonlocal overflow
        if status:
            overflow = True
        try:
            frames.put_nowait((time.monotonic(), bytes(data)))
        except queue.Full:
            overflow = True

    gain = InputGain(config.get('input_max_gain', 12.0), config.get('input_target_dbfs', -26.0),
                     config.get('input_floor_dbfs', -65.0))
    vad = webrtcvad.Vad(1)
    wake = WakeWord(config.get('wake_threshold', 0.3),
                    confirmations=config.get('wake_confirmations', 2),
                    cooldown_seconds=config.get('wake_cooldown_seconds', 1.5),
                    candidate_threshold=config.get('wake_candidate_threshold', 0.1))
    pre = deque(maxlen=150)  # 3 s includes the full wake phrase even on delayed detection
    active = False
    silence = total = 0
    chunk = []
    speech_votes = deque(maxlen=5)
    started = time.monotonic()
    previous_deadline = 0.0
    with sd.RawInputStream(device=index, samplerate=RATE, channels=1, dtype='int16',
                           blocksize=FRAME, callback=callback):
        log.info('LISTENING: %s; openWakeWord Hey Jarvis (threshold %.2f).', device['name'], wake.threshold)
        while not stop_event or not stop_event.is_set():
            if duration and time.monotonic() - started >= duration:
                break
            try:
                captured, frame = frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if overflow or time.monotonic() - captured > 4:
                log.warning('Audio fell behind; discarding the incomplete command')
                overflow = False
                while not frames.empty():
                    frames.get_nowait()
                active = False
                chunk, silence, total = [], 0, 0
                pre.clear()
                speech_votes.clear()
                conversation.pending = None
                plain_cancel_candidate()
                asr.reset()
                wake.reset()
                continue
            frame = gain.process(frame)
            pre.append(frame)
            # Keep acoustic context warm even while transcribing or awaiting
            # continuation. A new wake phrase starts a clean command session.
            score = wake.feed(frame)
            if score:
                log.info('openWakeWord %s: %.3f', 'candidate' if wake.needs_verification else 'detected Hey Jarvis', score)
                conversation.activate(captured, tentative=wake.needs_verification)
                previous_deadline = conversation.deadline
                asr.reset()
                active = True
                chunk = list(pre)
                silence = total = 0
                speech_votes.clear()
                continue
            if not active:
                conversation.expire(captured)
                if conversation.pending is None:
                    continue
            with timed('vad.is_speech', detail=True, audio_ms=20):
                speech = vad.is_speech(frame, RATE)
            speech_votes.append(speech)
            if not active:
                if sum(speech_votes) < 3:
                    conversation.expire(captured)
                    continue
                active = True
                asr.reset()
                previous_deadline = conversation.deadline
                chunk = list(pre)[-15:]
                silence = total = 0
            else:
                chunk.append(frame)
            total += 0.02
            # Command speech may last longer than the between-utterance timeout.
            conversation.deadline = captured + config['continuation_seconds']
            silence = 0 if speech else silence + 0.02
            final = silence >= config['pause_seconds']
            too_long = total >= config['max_utterance_seconds']
            if final or too_long or len(chunk) * 0.02 >= config['chunk_seconds']:
                log.info('AUDIO queue_lag=%.1f ms queued_frames=%d',
                         (time.monotonic() - captured) * 1000, frames.qsize())
                samples = np.frombuffer(b''.join(chunk), dtype=np.int16).astype(np.float32) / 32768
                chunk = []
                text = asr.feed(samples, final=final)
                conversation.confirm_wake(text)
                if final or too_long:
                    log.info('Heard: %s', text)
                    if not text.strip():
                        conversation.deadline = previous_deadline
                    if final and not too_long:
                        try:
                            with timed('command.plan_and_deliver'):
                                conversation.utterance(text, captured)
                        except Exception:
                            # Do not retry a possibly delivered POST.
                            conversation.pending = None
                            log.exception('Command failed; say the wake phrase again to start a new request')
                    else:
                        conversation.pending = None
                        plain_cancel_candidate()
                        log.warning('Utterance limit reached; command discarded')
                    active = False
                    speech_votes.clear()
