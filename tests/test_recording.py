import numpy as np
import types

import recording


class DummyStream:
    def __init__(self):
        self.call = 0

    def read(self, chunk):
        self.call += 1
        if self.call <= 2:
            return (np.ones(recording.CHUNK, dtype=np.int16) * 100).tobytes()
        else:
            return (np.zeros(recording.CHUNK, dtype=np.int16)).tobytes()

    def stop_stream(self):
        pass

    def close(self):
        pass


class DummyPyAudio:
    def open(self, **kwargs):
        return DummyStream()

    def terminate(self):
        pass


def test_record_audio_returns_data(monkeypatch):
    monkeypatch.setattr(recording, "SILENCE_FRAME_LIMIT", 2)
    monkeypatch.setattr(
        recording,
        "pyaudio",
        types.SimpleNamespace(PyAudio=lambda: DummyPyAudio(), paInt16=recording.FORMAT),
    )

    data = recording.record_audio()
    assert data is not None
    assert len(data) > 0
