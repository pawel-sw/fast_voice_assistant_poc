"""Windows microphone + CPU wake detector; all other inference runs on the VM."""
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
import uuid
from collections import deque

import sounddevice as sd
from websockets.sync.client import connect
import webrtcvad
from .remote_rpc import connect_remote, ready, RemotePlanner, SpeechOutput
from .client_brain import ClientBrain
from .control import Catalog, OpenHAB

from .input_audio import InputGain
from .wake import WakeWord
from . import observability as obs
from .timers import TimerService, parse_timer

log = logging.getLogger(__name__)


def connection(config):
    return connect_remote(config)


def show(event):
    kind = event['type']
    if kind == 'transcript':
        obs.Transcript(event['utterance']).update(event['text'], final=event.get('final', False))
        if event.get('final'):
            log.info('Heard: %s', event['text'])
    elif kind == 'confirmed':
        obs.plain_confirm_candidate()
    elif kind == 'command':
        status = {key: event[key] for key in ('dry_run', 'error') if event.get(key)}
        obs.plain_command(event['item'], event['command'], **status)
        log.info('REMOTE COMMAND %s <- %s%s', event['item'], event['command'], ' (dry run)' if event.get('dry_run') else '')
    elif kind == 'result':
        value = {'error': event['error']} if event.get('error') else {'result': event['result']}
        obs.plain_result(event['item'], event['location'], **value)
        log.info('TEMPERATURE %s: %s', event['location'], event.get('result', event.get('error')))
    elif kind == 'timing':
        logging.getLogger('jarvis.timing').info('REMOTE %s', event['message'])
    elif kind == 'error':
        log.error('VM: %s', event['message'])
    elif kind == 'done':
        obs.plain_cancel_candidate()


def text_request(config, text, execute=False):
    if execute and parse_timer(text) is not None:
        raise ValueError('Use the running voice listener for timers; --text supports timer dry-runs only')
    api = OpenHAB(config['openhab_url'])
    catalog = Catalog(api.items(), config)
    speaker = SpeechOutput(config)
    # CLI dry-runs use an isolated timer store and never publish or ring.
    timer = TimerService(lambda value: None, lambda cancel: None,
                         path=Path(__file__).resolve().parents[1]/'data/timer-text-test.json', start_worker=False)
    brain = ClientBrain(api, catalog, RemotePlanner(catalog, config), speaker, config, dry_run=not execute, timer=timer)
    try:
        session = brain.start()
        brain.transcript({'session': session, 'type': 'transcript', 'utterance': 0, 'text': text, 'final': True})
        while brain.listening or any((brain.current.plan, brain.current.delivery, brain.current.speech)):
            brain.tick()
            time.sleep(.02)
    finally:
        brain.close()
        timer.close()


def microphone(name):
    if name is None or isinstance(name, int):
        index = sd.default.device[0] if name is None else name
        device = sd.query_devices(index, 'input')
        sd.check_input_settings(device=index, channels=1, samplerate=16000, dtype='int16')
        return index, device
    devices = sd.query_devices()
    matches = [(i, d) for i, d in enumerate(devices) if name.lower() in d['name'].lower() and d['max_input_channels']]
    matches.sort(key=lambda entry: sd.query_hostapis(entry[1]['hostapi'])['name'] != 'MME')
    for index, device in matches:
        try:
            sd.check_input_settings(device=index, channels=1, samplerate=16000, dtype='int16')
            return index, device
        except sd.PortAudioError:
            pass
    raise RuntimeError(f'Microphone {name!r} unavailable')


def listen(config, duration=None, dry_run=False):
    api = OpenHAB(config['openhab_url'])
    catalog = Catalog(api.items(), config)
    speaker = SpeechOutput(config)
    # A separate HTTP session keeps countdown publishing independent of commands.
    timer_api = OpenHAB(config['openhab_url'])
    timer = TimerService(timer_api.timer_state if not dry_run else lambda value: None,
                         speaker.chime if not dry_run else lambda cancel: None,
                         path=config.get('timer_state_path'), start_worker=not dry_run)
    brain = ClientBrain(api, catalog, RemotePlanner(catalog, config), speaker, config, dry_run, timer=timer)
    index, device = microphone(config['microphone'])
    wake = WakeWord(config['wake_threshold'], confirmations=config['wake_confirmations'],
                    cooldown_seconds=config['wake_cooldown_seconds'], candidate_threshold=config['wake_candidate_threshold'])
    gain = InputGain(config['input_max_gain'], config['input_target_dbfs'], config['input_floor_dbfs'])
    vad = webrtcvad.Vad(1)
    frames = queue.Queue(maxsize=200)
    overflow = threading.Event()

    def callback(data, count, timing, status):
        if status:
            overflow.set()
        try:
            captured = time.monotonic()
            # PortAudio's ADC time identifies the end of the captured frame,
            # rather than the later time at which the main loop processes it.
            if timing is not None and timing.inputBufferAdcTime > 0:
                captured += timing.inputBufferAdcTime + count / 16000 - timing.currentTime
            frames.put_nowait((captured, bytes(data)))
        except queue.Full:
            overflow.set()

    started = time.monotonic()
    last_warning = 0
    try:
        with sd.RawInputStream(device=index, samplerate=16000, channels=1, dtype='int16', blocksize=320, callback=callback):
            while not duration or time.monotonic()-started < duration:
                try:
                    with connection(config) as ws:
                        ready(ws)
                        events = queue.Queue()
                        disconnected = threading.Event()

                        def receive(socket=ws, inbox=events, ended=disconnected):
                            try:
                                for message in socket:
                                    event = json.loads(message)
                                    if event.get('type') == 'transcript' and event.get('final'):
                                        obs.timestamp('final_transcript_received', event.get('session'), utterance=event.get('utterance'))
                                    inbox.put(event)
                            except Exception as exc:
                                inbox.put({'type': 'connection_error', 'message': str(exc)})
                            finally:
                                ended.set()

                        worker = threading.Thread(target=receive, daemon=True)
                        worker.start()
                        log.info('LISTENING: %s; client control; VM R2T2/Needle/Kokoro; speaker=%s', device['name'], config.get('speaker_device', 'default'))
                        pre, votes = deque(maxlen=150), deque(maxlen=5)
                        recording = awaiting = muted = False
                        stream_session = None
                        utterance = 0
                        chunk = []
                        silence = total = 0
                        last_speech = last_sent = None
                        last_progress = time.monotonic()
                        wake.reset()
                        while not duration or time.monotonic()-started < duration:
                            if disconnected.is_set():
                                raise ConnectionError('VM disconnected')
                            while not events.empty():
                                event = events.get_nowait()
                                if event['type'] == 'timing':
                                    show(event)
                                elif event['type'] == 'transcript':
                                    if brain.current and event.get('session') == brain.current.id:
                                        last_progress = time.monotonic()
                                        if event.get('final'):
                                            awaiting = False
                                        brain.transcript(event)
                                elif event['type'] == 'error':
                                    log.error('VM: %s', event['message'])
                                    brain.cancel()
                            brain.tick(input_active=recording or awaiting)
                            if stream_session and not brain.listening:
                                ws.send(json.dumps({'type': 'cancel'}))
                                stream_session = None
                                recording = awaiting = False
                                chunk = []
                            if stream_session and time.monotonic()-last_progress > 15:
                                brain.cancel()
                                raise TimeoutError('No transcription progress for 15 seconds')
                            try:
                                captured, frame = frames.get(timeout=.2)
                            except queue.Empty:
                                continue
                            if speaker.muted:
                                if not muted:
                                    wake.reset()
                                    pre.clear()
                                    votes.clear()
                                muted = True
                                continue
                            if muted:
                                muted = False
                                wake.reset()
                                pre.clear()
                                votes.clear()
                            if overflow.is_set() or time.monotonic()-captured > 2:
                                overflow.clear()
                                while not frames.empty():
                                    frames.get_nowait()
                                brain.cancel()
                                pre.clear()
                                wake.reset()
                                continue
                            frame = gain.process(frame)
                            pre.append(frame)
                            score = wake.feed(frame)
                            speech = vad.is_speech(frame, 16000)
                            votes.append(speech)
                            if speech:
                                last_speech = captured
                            timer_interrupt = timer.is_ringing and not brain.listening and sum(votes) >= 3
                            if score or timer_interrupt:
                                stream_session = brain.start(tentative=wake.needs_verification if score else False,
                                                             timer_only=not bool(score))
                                utterance = 0
                                if score:
                                    obs.timestamp('wake_detected', stream_session, tentative=wake.needs_verification)
                                ws.send(json.dumps({'type': 'asr_start', 'session': stream_session, 'utterance': utterance}))
                                ws.send(b''.join(pre))
                                last_sent = time.monotonic()
                                recording, awaiting = True, False
                                silence = total = 0
                                chunk = []
                                last_progress = time.monotonic()
                                if score:
                                    log.info('Wake %s %s %.3f', wake.last_keyword, 'candidate' if wake.needs_verification else 'detected', score)
                                else:
                                    log.info('Listening for timer stop')
                                continue
                            if not brain.listening:
                                continue
                            if not recording:
                                if awaiting or sum(votes) < 3:
                                    continue
                                utterance += 1
                                ws.send(json.dumps({'type': 'asr_start', 'session': stream_session, 'utterance': utterance}))
                                chunk = list(pre)[-15:]
                                recording = True
                                silence = total = 0
                                last_progress = time.monotonic()
                            else:
                                chunk.append(frame)
                            total += .02
                            silence = 0 if speech else silence+.02
                            if total >= config['max_utterance_seconds']:
                                brain.cancel()
                                log.warning('Utterance limit reached; discarded')
                                continue
                            final = silence >= config['pause_seconds']
                            if final:
                                obs.timestamp('vad_endpoint', stream_session, utterance=utterance)
                                if last_speech is not None:
                                    obs.timestamp('last_speech_audio', stream_session, at=last_speech, utterance=utterance, source='audio_frame_end')
                            if final or len(chunk)*.02 >= config['chunk_seconds']:
                                if chunk:
                                    ws.send(b''.join(chunk))
                                    last_sent = time.monotonic()
                                    chunk = []
                                if final:
                                    obs.timestamp('last_audio_sent_to_r2t2', stream_session, at=last_sent, utterance=utterance)
                                    ws.send(json.dumps({'type': 'asr_finish'}))
                                    recording, awaiting = False, True
                                    votes.clear()
                except KeyboardInterrupt:
                    return
                except Exception as exc:
                    brain.cancel()
                    if time.monotonic()-last_warning > 30:
                        log.warning('Remote unavailable (%s); reconnecting without replay', exc)
                        last_warning = time.monotonic()
                    time.sleep(3)
    finally:
        brain.close()
        timer.close()
