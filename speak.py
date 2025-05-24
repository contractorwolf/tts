import sys
import time
import torch
from TTS.api import TTS
import sounddevice as sd

# determine if GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TEXT = "the journey of a thousand miles begins with a single step"

print("Loading TTS model...")
# display device and library information
print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"TTS version: {getattr(TTS, '__version__', 'unknown')}")
if DEVICE == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# load the speedy-speech model once and time how long it takes
load_start = time.time()
tts = TTS(model_name="tts_models/en/ljspeech/speedy-speech").to(DEVICE)
print(f"Model loaded in {(time.time() - load_start) * 1000:.0f} ms")


def speak(text: str) -> None:
    """Convert text to speech and play it."""
    print(f"Speaking: {text}")
    gen_start = time.time()
    wav = tts.tts(text=text)
    print(f"Generation took {(time.time() - gen_start) * 1000:.0f} ms")
    play_start = time.time()
    sd.play(wav, samplerate=22050)
    sd.wait()
    print(f"Playback took {(time.time() - play_start) * 1000:.0f} ms")


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    speak(text)
    print("Type text and press enter (Ctrl+C to exit).")
    try:
        while True:
            user_text = input("> ")
            if user_text.strip():
                speak(user_text)
    except KeyboardInterrupt:
        print("\nExiting...")
