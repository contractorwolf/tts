import torch
from pydub import AudioSegment
from pydub.playback import play
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from datasets import load_dataset

processor2 = None
model2 = None
speaker_embeddings = None
vocoder = None


def load_speech_models():
    global processor2, model2, speaker_embeddings, vocoder
    if processor2 is None:
        processor2 = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model2 = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


def speak(text):
    """Convert text to speech and play it."""
    load_speech_models()

    inputs = processor2(text=text, return_tensors="pt")
    speech = model2.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    audio = AudioSegment(
        speech.numpy().tobytes(),
        frame_rate=16000,
        sample_width=speech.numpy().dtype.itemsize,
        channels=1,
    )
    play(audio)
