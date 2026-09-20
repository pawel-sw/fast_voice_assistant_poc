"""openWakeWord ONNX wake detector; accepts the microphone's 20 ms PCM frames."""
from pathlib import Path

import numpy as np

from .observability import timed

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'openwakeword'
WAKE_MODELS = ('hey_jarvis', 'alexa')


def download_models():
    from openwakeword.utils import download_models as download
    download(model_names=list(WAKE_MODELS), target_directory=str(MODEL_DIR))


class WakeWord:
    def __init__(self, threshold=0.3, model=None, confirmations=2, cooldown_seconds=1.5, candidate_threshold=0.1):
        if not 0 < threshold <= 1:
            raise ValueError('Wake threshold must be in (0, 1]')
        if model is None:
            from openwakeword.model import Model
            paths = [MODEL_DIR / f'{name}_v0.1.onnx' for name in WAKE_MODELS]
            if not all(path.exists() for path in paths):
                raise RuntimeError('Missing wake models. Run python -m jarvis.wake or setup.ps1.')
            with timed('openwakeword.load'):
                model = Model(wakeword_models=[str(path) for path in paths], inference_framework='onnx',
                          melspec_model_path=str(MODEL_DIR / 'melspectrogram.onnx'),
                          embedding_model_path=str(MODEL_DIR / 'embedding_model.onnx'))
        self.model = model
        self.threshold = threshold
        if not 0 < candidate_threshold <= threshold:
            raise ValueError('Candidate threshold must be positive and no higher than wake threshold')
        self.candidate_threshold = candidate_threshold
        self.needs_verification = False
        if confirmations < 1 or cooldown_seconds < 0:
            raise ValueError('Invalid wake confirmation or cooldown setting')
        self.confirmations = confirmations
        self.cooldown_frames = round(cooldown_seconds / 0.08)
        self.buffer = bytearray()
        self.hits = self.quiet = 0
        self.cooldown = 0
        self.armed = True
        self.last_score = 0.0
        self.hit_keyword = None
        self.last_keyword = None

    def reset(self):
        self.buffer.clear()
        self.model.reset()
        self.hits = self.quiet = self.cooldown = 0
        self.armed = True
        self.hit_keyword = self.last_keyword = None

    def feed(self, frame):
        self.buffer.extend(frame)
        peak = 0.0
        # openWakeWord expects 80 ms / 1280 samples, signed PCM16 at 16 kHz.
        while len(self.buffer) >= 2560:
            samples = np.frombuffer(bytes(self.buffer[:2560]), dtype='<i2')
            del self.buffer[:2560]
            with timed('openwakeword.predict', detail=True, audio_ms=80):
                prediction = self.model.predict(samples)
            keyword, score = max(((name, float(value)) for name, value in prediction.items()),
                                 key=lambda item: item[1], default=(None, 0.0))
            self.last_score = score
            self.cooldown = max(0, self.cooldown - 1)
            if not self.armed:
                self.quiet = self.quiet + 1 if score < min(0.15, self.threshold / 2) else 0
                if self.quiet >= 4 and self.cooldown == 0:
                    self.armed = True
                    self.hits = 0
                continue
            self.hits = (self.hits + 1 if keyword == self.hit_keyword else 1) if score >= self.threshold else 0
            self.hit_keyword = keyword
            if self.hits >= self.confirmations:
                peak = max(peak, score)
                self.needs_verification = False
                self.last_keyword = keyword
                self.armed = False
                self.hits = self.quiet = 0
                self.cooldown = self.cooldown_frames
            elif self.candidate_threshold <= score < self.threshold:
                # A brief, coarticulated wake phrase may never reach the normal
                # threshold. This event cannot authorize tools until ASR agrees.
                peak = max(peak, score)
                self.needs_verification = True
                self.last_keyword = keyword
                self.armed = False
                self.hits = self.quiet = 0
                self.cooldown = self.cooldown_frames
        return peak


if __name__ == '__main__':
    download_models()
