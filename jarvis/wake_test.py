"""Read-only microphone test: no ASR, device calls, or saved microphone audio.

Run after stop.ps1: python -m jarvis.wake_test --seconds 20
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import sounddevice as sd

from .audio import microphone
from .input_audio import InputGain
from .instance import listener_lock
from .wake import WakeWord


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'config.json').read_text())
    with listener_lock(root / 'data/listener.lock'):
        wake = WakeWord(cfg['wake_threshold'], confirmations=cfg['wake_confirmations'], cooldown_seconds=cfg['wake_cooldown_seconds'],
                        candidate_threshold=cfg.get('wake_candidate_threshold', 0.1))
        gain = InputGain(cfg['input_max_gain'], cfg['input_target_dbfs'], cfg['input_floor_dbfs'])
        index, device = microphone(cfg['microphone'])
        print('Say Hey Jarvis at your usual quiet volume. No device actions or recording.', flush=True)
        started = report = time.monotonic()
        peak_score, peak_rms = 0.0, 0.0
        with sd.RawInputStream(device=index, samplerate=16000, channels=1, dtype='int16', blocksize=320) as stream:
            while time.monotonic() - started < args.seconds:
                frame, overflow = stream.read(320)
                score = wake.feed(gain.process(bytes(frame)))
                peak_score, peak_rms = max(peak_score, wake.last_score), max(peak_rms, gain.rms)
                if score:
                    label = 'CANDIDATE (requires ASR verification)' if wake.needs_verification else 'DETECTED'
                    print(f'{label} score={score:.3f}', flush=True)
                if time.monotonic() - report >= 1:
                    print(f'input={20*np.log10(max(peak_rms,1e-9)):.1f} dBFS gain={gain.gain:.1f}x wake_peak={peak_score:.3f}', flush=True)
                    report, peak_score, peak_rms = time.monotonic(), 0.0, 0.0


if __name__ == '__main__':
    main()
