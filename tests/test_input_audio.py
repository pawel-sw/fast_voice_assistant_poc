import numpy as np

from jarvis.input_audio import InputGain


def test_quiet_signal_boosted_but_silence_unchanged():
    gain = InputGain()
    quiet = (np.sin(np.arange(320) * .17) * 100).astype('<i2').tobytes()
    for _ in range(10):
        boosted = gain.process(quiet)
    assert np.max(np.abs(np.frombuffer(boosted, '<i2'))) > 1000
    silence = bytes(640)
    assert gain.process(silence) == silence
    noise = np.ones(320, dtype='<i2').tobytes()
    assert gain.process(noise) == noise


def test_loud_sound_after_quiet_speech_does_not_clip():
    gain = InputGain()
    for _ in range(20):
        gain.process(np.full(320, 100, dtype='<i2').tobytes())
    loud = (np.sin(np.arange(320) * .17) * 25000).astype('<i2')
    output = np.frombuffer(gain.process(loud.tobytes()), '<i2')
    assert np.max(np.abs(output)) < 32767
