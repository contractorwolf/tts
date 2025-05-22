import types
import torch

import speech_synthesis


def test_speak(monkeypatch):
    outputs = []

    dummy_processor = types.SimpleNamespace(
        __call__=lambda text, return_tensors: {"input_ids": torch.tensor([[0]])}
    )
    dummy_model = types.SimpleNamespace(
        generate_speech=lambda ids, speaker_embeddings, vocoder: torch.zeros(16000)
    )
    dummy_embeddings = torch.zeros(1)
    dummy_vocoder = object()

    monkeypatch.setattr(speech_synthesis, "processor2", dummy_processor)
    monkeypatch.setattr(speech_synthesis, "model2", dummy_model)
    monkeypatch.setattr(speech_synthesis, "speaker_embeddings", dummy_embeddings)
    monkeypatch.setattr(speech_synthesis, "vocoder", dummy_vocoder)
    monkeypatch.setattr(speech_synthesis, "load_speech_models", lambda: None)

    class DummyAudioSegment:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(speech_synthesis, "AudioSegment", DummyAudioSegment)
    monkeypatch.setattr(speech_synthesis, "play", lambda audio: outputs.append(True))

    speech_synthesis.speak("hello")
    assert outputs == [True]
