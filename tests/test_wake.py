import numpy as np

from jarvis.wake import WakeWord


class Model:
    def __init__(self):
        self.frames = []
        self.resets = 0

    def predict(self, samples):
        self.frames.append(samples)
        return {'hey_jarvis': 0.7}

    def reset(self):
        self.resets += 1


def test_20ms_frames_aggregate_without_loss():
    model = Model()
    wake = WakeWord(model=model, confirmations=1)
    samples = np.arange(1280, dtype=np.int16)
    for frame in np.split(samples, 4)[:3]:
        assert wake.feed(frame.tobytes()) == 0
    assert wake.feed(samples[960:].tobytes()) == 0.7
    np.testing.assert_array_equal(model.frames[0], samples)


def test_threshold_and_reset_discard_partial_audio():
    model = Model()
    wake = WakeWord(0.8, model, candidate_threshold=0.8)
    assert wake.feed(bytes(2560)) == 0
    wake.feed(bytes(640))
    wake.reset()
    assert not wake.buffer
    assert model.resets == 1


def test_confirmation_debounce_and_rearm():
    class Scores:
        score = 0.45
        def predict(self, samples):
            return {'hey_jarvis': self.score}
    model = Scores()
    wake = WakeWord(model=model)
    assert wake.feed(bytes(2560)) == 0
    assert wake.feed(bytes(2560)) == 0.45
    assert all(wake.feed(bytes(2560)) == 0 for _ in range(25))
    model.score = 0.05
    assert all(wake.feed(bytes(2560)) == 0 for _ in range(4))
    model.score = 0.45
    assert wake.feed(bytes(2560)) == 0
    assert wake.feed(bytes(2560)) == 0.45


def test_one_noisy_frame_cannot_activate():
    model = Model()
    wake = WakeWord(model=model)
    assert wake.feed(bytes(2560)) == 0
    model.predict = lambda samples: {'hey_jarvis': 0.01}
    assert wake.feed(bytes(2560)) == 0
    assert wake.armed


def test_low_score_produces_only_a_candidate():
    model = Model()
    model.predict = lambda samples: {'hey_jarvis': 0.15}
    wake = WakeWord(model=model)
    assert wake.feed(bytes(2560)) == 0.15
    assert wake.needs_verification
    assert wake.feed(bytes(2560)) == 0


def test_alexa_detection_and_no_cross_model_confirmation():
    model = Model()
    scores = iter([{'hey_jarvis': .7, 'alexa': .01},
                   {'hey_jarvis': .01, 'alexa': .7},
                   {'hey_jarvis': .01, 'alexa': .7}])
    model.predict = lambda samples: next(scores)
    wake = WakeWord(model=model)
    assert wake.feed(bytes(2560)) == 0
    assert wake.feed(bytes(2560)) == 0
    assert wake.feed(bytes(2560)) == .7
    assert wake.last_keyword == 'alexa'
    assert not wake.needs_verification
