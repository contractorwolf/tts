import numpy as np
import types
import torch

import transcription


def test_process_audio_returns_text(monkeypatch):
    dummy_processor = types.SimpleNamespace(
        __call__=lambda audio_input, return_tensors, sampling_rate: types.SimpleNamespace(
            input_values=torch.tensor([[0.0]])
        ),
        batch_decode=lambda ids: ["hello"],
    )
    dummy_model = types.SimpleNamespace(
        __call__=lambda input_values: types.SimpleNamespace(logits=torch.zeros(1, 1, 1))
    )

    monkeypatch.setattr(transcription, "processor", dummy_processor)
    monkeypatch.setattr(transcription, "model", dummy_model)

    audio = np.ones(16000, dtype=np.int16)
    result = transcription.process_audio(audio)
    assert result == "hello"
