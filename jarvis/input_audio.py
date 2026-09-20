"""Bounded gain for quiet microphone speech, without amplifying digital silence."""
import numpy as np


class InputGain:
    def __init__(self, max_gain=12.0, target_dbfs=-26.0, floor_dbfs=-65.0):
        if not 1 <= max_gain <= 32 or not -50 <= target_dbfs <= -10 or floor_dbfs >= target_dbfs:
            raise ValueError('Invalid input gain settings')
        self.maximum = max_gain
        self.target = 10 ** (target_dbfs / 20)
        self.floor = 10 ** (floor_dbfs / 20)
        self.gain = 1.0
        self.rms = 0.0

    def process(self, frame):
        samples = np.frombuffer(frame, dtype='<i2').astype(np.float32) / 32768
        self.rms = float(np.sqrt(np.mean(samples * samples)))
        peak = float(np.max(np.abs(samples)))
        if self.rms < self.floor:
            # Do not lift silence/hiss into VAD's speech range.
            self.gain = 1.0
            return frame
        desired = min(self.maximum, max(1.0, self.target / self.rms))
        self.gain += 0.35 * (desired - self.gain)
        # Immediate limiter when a loud sound follows quiet speech.
        self.gain = min(self.gain, max(1.0, 0.9 / max(peak, 1e-9)))
        return np.clip(np.rint(samples * self.gain * 32768), -32768, 32767).astype('<i2').tobytes()
