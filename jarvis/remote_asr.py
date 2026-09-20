"""GPU R2T2 streaming backend, loaded only by the Ubuntu service."""
import numpy as np
from .observability import timed


class RemoteASR:
    def __init__(self, config):
        from r2t2 import R2T2ASRModel
        with timed('r2t2.vllm.load'):
            self.model = R2T2ASRModel.LLM(
                model=config.get('server_asr_model', 'models/R2T2'),
                dtype='float16', gpu_memory_utilization=config.get('asr_gpu_memory_utilization', 0.70),
                max_model_len=2048, max_num_seqs=1, enforce_eager=True,
                max_new_tokens=64, enable_prefix_caching=False,
            )
        self.chunk_seconds = config['chunk_seconds']
        # JIT/feature-extractor initialization must finish before /health is ready.
        with timed('r2t2.vllm.warmup'):
            self.feed(np.zeros(round(16000*self.chunk_seconds), dtype=np.float32), self.new_state(), final=True)

    def new_state(self):
        return self.model.init_streaming_state(
            language='English', chunk_size_sec=self.chunk_seconds,
            unfixed_chunk_num=0, unfixed_token_num=1)

    def feed(self, samples, state, final=False):
        with timed('r2t2.vllm.stream', detail=True, audio_ms=round(len(samples)/16)):
            self.model.streaming_transcribe(samples, state, max_new_tokens=16)
        if final:
            # Flush even when the last frame landed exactly on a chunk boundary.
            if not state.buffer.size:
                state.buffer = np.zeros(160, dtype=np.float32)
            with timed('r2t2.vllm.finish'):
                self.model.finish_streaming_transcribe(state, max_new_tokens=64)
        return state.text
