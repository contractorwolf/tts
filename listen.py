import torch
import numpy as np
import pyaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RATE = 16000
CHUNK = 1024
SILENCE_FRAMES = int(RATE / CHUNK * 1.5)  # ~1.5s of silence
THRESHOLD = 25

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)


def listen(stream):
    """Record audio until consecutive silence then return transcription."""
    frames, silent = [], 0
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = np.sqrt(np.mean(np.frombuffer(data, np.int16) ** 2))  # loudness
        silent = silent + 1 if rms < THRESHOLD else 0
        if silent >= SILENCE_FRAMES:
            break
    audio = np.frombuffer(b"".join(frames), np.int16)
    if np.sqrt(np.mean(audio ** 2)) <= THRESHOLD:
        return None
    inputs = processor(audio, sampling_rate=RATE, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        ids = model(inputs).logits.argmax(dim=-1)
    return processor.batch_decode(ids)[0]


def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Listening... (Ctrl+C to stop)")
    try:
        while True:
            text = listen(stream)
            if text:
                print(text.lower())
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
