import asyncio
import numpy as np
import pyaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio settings
RATE = 16000
CHUNK = 1024
SILENCE_FRAMES = int(RATE / CHUNK * 1.5)  # ~1.5s of silence
THRESHOLD = 30

# Load models
print("Loading models...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(
    DEVICE, dtype=torch.float32
)
print("Model loaded")


def record(stream: pyaudio.Stream) -> np.ndarray | None:
    """Record until silence and return the audio."""
    frames, silent = [], 0
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = np.sqrt(np.mean(np.frombuffer(data, np.int16) ** 2))
        silent = silent + 1 if rms < THRESHOLD else 0
        if silent >= SILENCE_FRAMES:
            break
    audio = np.frombuffer(b"".join(frames), np.int16)
    return audio if np.sqrt(np.mean(audio ** 2)) > THRESHOLD else None


def transcribe(audio: np.ndarray) -> str:
    """Return text transcription for ``audio``."""
    inputs = processor(audio, sampling_rate=RATE, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        ids = model(inputs).logits.argmax(dim=-1)
    return processor.batch_decode(ids)[0]


ASYNC_SLEEP = 0.1

async def producer(buf: list[np.ndarray]) -> None:
    """Capture microphone audio."""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    try:
        while True:
            audio = record(stream)
            if audio is not None:
                buf.append(audio)
            await asyncio.sleep(ASYNC_SLEEP)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


async def consumer(buf: list[np.ndarray]) -> None:
    """Transcribe recorded audio."""
    while True:
        if buf:
            audio = buf.pop(0)
            text = await asyncio.get_event_loop().run_in_executor(None, transcribe, audio)
            print("Transcribed:", text.lower())
        await asyncio.sleep(ASYNC_SLEEP)


async def main() -> None:
    print("Listening...")
    buf: list[np.ndarray] = []
    await asyncio.gather(producer(buf), consumer(buf))


if __name__ == "__main__":
    asyncio.run(main())
