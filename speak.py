import sys
import torch
from TTS.api import TTS
import sounddevice as sd

# Available speech models (from main.py)
# 10: tts_models/en/ek1/tacotron2                           doesnt work
# 11: tts_models/en/ljspeech/tacotron2-DDC                   doesnt work
# 12: tts_models/en/ljspeech/tacotron2-DDC_ph
# 13: tts_models/en/ljspeech/glow-tts
# 14: tts_models/en/ljspeech/speedy-speech
# 15: tts_models/en/ljspeech/tacotron2-DCA
# 16: tts_models/en/ljspeech/vits
# 17: tts_models/en/ljspeech/vits--neon
# 18: tts_models/en/ljspeech/fast_pitch
# 19: tts_models/en/ljspeech/overflow
# 20: tts_models/en/ljspeech/neural_hmm
# 21: tts_models/en/vctk/vits
# 22: tts_models/en/vctk/fast_pitch
# 23: tts_models/en/sam/tacotron-DDC
# 24: tts_models/en/blizzard2013/capacitron-t2-c50
# 25: tts_models/en/blizzard2013/capacitron-t2-c150_v2
# 26: tts_models/en/multi-dataset/tortoise-v2
# 27: tts_models/en/jenny/jenny

#speech_model = "tts_models/en/ljspeech/speedy-speech" #"tts_models/en/ljspeech/speedy-speech"
speech_model = "tts_models/en/ljspeech/vits"  # Changed from speedy-speech to vits

# determine if GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TEXT = "the journey of a thousand miles begins with a single step"
# Minimum number of characters for each utterance.  Shorter phrases cause
# errors in some models, so they are padded with spaces up to this length.
MIN_INPUT_LEN = 20



print("Loading TTS model...")
# load the speedy-speech model once
tts = TTS(model_name=speech_model).to(DEVICE)


def _prepare_text(text: str) -> str:
    """Pad short text inputs with natural pauses so the model receives enough characters."""
    text = text.strip()
    if len(text) < MIN_INPUT_LEN:
        # Option 1: Add ellipses (creates natural pauses)
        padding_needed = MIN_INPUT_LEN - len(text)
        text = text + "..." * (padding_needed // 3 + 1)
        
        # Option 2: Add commas with spaces (alternative)
        # text = text + ", " * (padding_needed // 2 + 1)
        
        # Option 3: SSML break tags (if model supports SSML)
        # text = text + '<break time="0.5s"/>' * (padding_needed // 20 + 1)
        
    return text[:MIN_INPUT_LEN]  # Trim to exact length if we over-padded


def speak(text: str) -> None:
    """Convert text to speech and play it."""
    text = _prepare_text(text)
    print(f"Speaking: {text}")
    wav = tts.tts(text=text)
    sd.play(wav, samplerate=22050)
    sd.wait()


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
