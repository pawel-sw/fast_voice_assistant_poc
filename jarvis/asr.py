"""Native CUDA adaptation of R2T2's accumulated-audio/prefix streaming algorithm.

Reference: netease-youdao/Confucius4-R2T2 r2t2/r2t2_asr.py (Apache-2.0).
The upstream streaming API is vLLM-only. This adapter uses its same checkpoint
with the Qwen Transformers processor and reuses all but the last decoded token.
"""
import logging
from pathlib import Path

import numpy as np

from .observability import Transcript, timed

log = logging.getLogger(__name__)


class R2T2:
    def __init__(self, model):
        import torch
        from qwen_asr import Qwen3ASRModel
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for R2T2. Install CUDA PyTorch using setup.ps1.')
        torch.set_num_threads(4)
        self.torch = torch
        local = Path(__file__).resolve().parents[1] / 'models' / 'R2T2'
        model = str(local) if (local / 'model.safetensors').exists() else model
        log.info('Loading R2T2 on %s', torch.cuda.get_device_name(0))
        with timed('r2t2.load'):
            self.asr = Qwen3ASRModel.from_pretrained(model, dtype=torch.bfloat16,
                                               device_map='cuda:0', attn_implementation='sdpa',
                                               max_new_tokens=64)
        self.prompt = self.asr._build_text_prompt(context='Hey Jarvis', force_language='English')
        self.reset()

    def reset(self):
        self.audio = np.empty(0, dtype=np.float32)
        self.raw = ''
        self.text = ''
        self.sequence = getattr(self, 'sequence', 0) + 1
        self.transcript = None

    def feed(self, samples, final=False):
        if self.transcript is None:
            self.transcript = Transcript(f'utterance-{self.sequence}')
        self.audio = np.concatenate((self.audio, np.asarray(samples, dtype=np.float32)))
        if self.audio.size < 1280:
            return self.text
        tokenizer = self.asr.processor.tokenizer
        ids = tokenizer.encode(self.raw, add_special_tokens=False)
        prefix = tokenizer.decode(ids[:-1]) if ids else ''
        while '\ufffd' in prefix and ids:
            ids = ids[:-1]
            prefix = tokenizer.decode(ids[:-1])
        from transformers import TextStreamer
        transcript = self.transcript

        class LiveStreamer(TextStreamer):
            emitted = ''

            def on_finalized_text(self, text, stream_end=False):
                if not text:
                    return
                self.emitted += text
                # TextStreamer emits words during generate(), not after it returns.
                live = (prefix + self.emitted).split('|')[0].replace('\ufffd', '').strip()
                transcript.update(live, provisional=True)

        streamer = LiveStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with timed('r2t2.feed', utterance=self.sequence, audio_ms=round(len(self.audio) / 16), final=final), self.torch.inference_mode():
            with timed('r2t2.preprocess', utterance=self.sequence):
                inputs = self.asr.processor(text=[self.prompt + prefix], audio=[self.audio],
                                        return_tensors='pt', padding=True)
                inputs = inputs.to(self.asr.model.device).to(self.asr.model.dtype)
                self.torch.cuda.synchronize()
            with timed('r2t2.generate', utterance=self.sequence):
                outputs = self.asr.model.generate(**inputs, max_new_tokens=64 if final else 12,
                                              do_sample=False, streamer=streamer,
                                              pad_token_id=tokenizer.eos_token_id)
                self.torch.cuda.synchronize()
            sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs
            generated = tokenizer.decode(sequences[0, inputs['input_ids'].shape[1]:],
                                          skip_special_tokens=True)
        self.raw = (prefix + generated).split('|')[0].replace('\ufffd', '')
        self.text = self.raw.strip()
        self.transcript.update(self.text, final=final)
        return self.text
