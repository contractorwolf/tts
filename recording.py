import asyncio
import numpy as np
import pyaudio

# Audio configuration constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_LIMIT = 1.5
SILENCE_FRAME_LIMIT = int((RATE // CHUNK) * SILENCE_LIMIT)
THRESHOLD = 25


def record_audio():
    """Record audio until silence is detected and return the data."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []
    silent_frames = 0

    while True:
        data = stream.read(CHUNK)
        current_frame = np.frombuffer(data, dtype=np.int16)
        frames.append(data)

        rms = np.sqrt(np.abs(np.mean(current_frame ** 2)))
        if rms < THRESHOLD:
            silent_frames += 1
        else:
            silent_frames = 0

        if silent_frames >= SILENCE_FRAME_LIMIT:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
    rms = np.sqrt(np.mean(audio_data ** 2))

    if rms > THRESHOLD:
        return audio_data
    return None


async def record_audio_async(buffer):
    try:
        while True:
            audio_data = record_audio()
            if audio_data is not None:
                buffer.append(audio_data)
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Error in record_audio_async: {str(e)}")
