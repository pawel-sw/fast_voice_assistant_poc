"""Server-side audio endpointing. No microphone or wake model dependencies."""
import logging
from collections import deque
import numpy as np
import webrtcvad
from .control import Conversation
from .observability import timed

log = logging.getLogger(__name__)


class AudioSession:
    def __init__(self, asr, planner, execute, config, emit, tentative=False):
        self.asr, self.config, self.emit = asr, config, emit
        self.conversation = Conversation(planner, execute, config['wake_phrase'], config['continuation_seconds'])
        self.conversation.activate(0, tentative=tentative)
        self.vad = webrtcvad.Vad(1)
        self.pre = deque(maxlen=15)
        self.votes = deque(maxlen=5)
        self.active = True
        self.elapsed = self.silence = self.total = 0
        self.chunk = []
        self.utterance = 0
        self.closed = False
        self.asr.reset()

    def feed(self, data, preroll=False):
        if len(data) % 640:
            raise ValueError('Audio must contain whole 20 ms PCM16 frames')
        for offset in range(0, len(data), 640):
            if self.closed:
                break
            frame = data[offset:offset+640]
            self.elapsed += .02
            self.pre.append(frame)
            speech = self.vad.is_speech(frame, 16000)
            self.votes.append(speech)
            if not self.active:
                self.conversation.expire(self.elapsed)
                if self.conversation.pending is None:
                    self.finish()
                    break
                if sum(self.votes) < 3:
                    continue
                self.active = True
                self.asr.reset()
                self.chunk = list(self.pre)
                self.silence = self.total = 0
                self.utterance += 1
            else:
                self.chunk.append(frame)
            self.total += .02
            self.conversation.deadline = self.elapsed + self.config['continuation_seconds']
            self.silence = 0 if speech else self.silence + .02
            # Preroll contains silence BEFORE the wake: never endpoint inside it.
            final = not preroll and self.silence >= self.config['pause_seconds']
            too_long = self.total >= self.config['max_utterance_seconds']
            if too_long:
                self.emit({'type': 'error', 'message': 'Utterance too long; discarded'})
                self.finish()
                break
            if final or len(self.chunk)*.02 >= self.config['chunk_seconds']:
                samples = np.frombuffer(b''.join(self.chunk), dtype='<i2').astype(np.float32)/32768
                self.chunk = []
                text = self.asr.feed(samples, final=final)
                was_verified = self.conversation.verified
                self.conversation.confirm_wake(text)
                if not was_verified and self.conversation.verified:
                    self.emit({'type': 'confirmed'})
                self.emit({'type': 'transcript', 'utterance': self.utterance, 'text': text, 'final': final})
                if final:
                    log.info('Heard: %s', text)
                    with timed('command.plan_and_deliver'):
                        self.conversation.utterance(text, self.elapsed)
                    self.active = False
                    self.votes.clear()
                    if self.conversation.pending is None:
                        self.finish()

    def finish(self):
        self.closed = True
        self.conversation.pending = None
        self.emit({'type': 'done'})
