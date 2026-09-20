"""VM-only Kokoro synthesis. Output is mono signed PCM16 at 24 kHz."""
import numpy as np
from .observability import timed


class Kokoro:
    rate = 24000

    def __init__(self, config):
        import torch
        from kokoro import KPipeline
        self.voice = config.get('tts_voice', 'af_heart')
        self.speed = config.get('tts_speed', 1.2)
        self.device = config.get('tts_device', 'cuda')
        torch.set_num_threads(4)
        with timed('kokoro.load'):
            self.pipeline = KPipeline(lang_code='a', device=self.device)
        with timed('kokoro.warmup'):
            list(self.chunks('Ready.'))

    def chunks(self, text, speed=None):
        with timed('kokoro.synthesize', detail=True, characters=len(text)):
            for _, _, audio in self.pipeline(text, voice=self.voice, speed=self.speed if speed is None else speed):
                pcm = (np.clip(audio.detach().cpu().numpy(), -1, 1)*32767).astype('<i2').tobytes()
                # Bounded 100 ms packets; playback can start before later sentences.
                for offset in range(0, len(pcm), 4800):
                    yield pcm[offset:offset+4800]
