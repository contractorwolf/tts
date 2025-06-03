# for: ljspeech/vits
# needed: sudo apt-get install espeak espeak-data
# needed: pip install espeak

# needed: pip install termios
# needed: pip install tty
# needed: pip install select

import sys
import torch
from TTS.api import TTS
import sounddevice as sd
import termios
import tty
# import select

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




# Add this near the top with other globals
AUDIO_CACHE = {}  # Cache for storing generated audio
MAX_CACHE_SIZE = 100  # Limit cache size to prevent memory issues

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
    
    # Check if audio is already cached
    if text in AUDIO_CACHE:
        print("(using cached audio)")
        wav = AUDIO_CACHE[text]
    else:
        # Generate new audio and cache it
        wav = tts.tts(text=text)
        
        # Manage cache size
        if len(AUDIO_CACHE) >= MAX_CACHE_SIZE:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(AUDIO_CACHE))
            del AUDIO_CACHE[oldest_key]
        
        AUDIO_CACHE[text] = wav
        print(f"(cached - total cached: {len(AUDIO_CACHE)})")
    
    sd.play(wav, samplerate=22050)
    sd.wait()


def get_char():
    """Get a single character from stdin without pressing enter"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def get_input_with_history(prompt, history, history_index):
    """Get input with arrow key history navigation"""
    print(prompt, end='', flush=True)
    current_text = ""
    cursor_pos = 0
    temp_history_index = history_index
    
    def refresh_line():
        """Clear current line and redraw prompt + current text"""
        print(f'\r\033[K{prompt}{current_text}', end='', flush=True)
    
    while True:
        char = get_char()
        
        if char == '\r' or char == '\n':  # Enter key
            print()
            return current_text, len(history)
        
        elif char == '\x03':  # Ctrl+C
            raise KeyboardInterrupt
        
        elif char == '\x1b':  # Escape sequence (arrow keys)
            next1, next2 = get_char(), get_char()
            if next1 == '[':
                if next2 == 'A':  # Up arrow
                    if history and temp_history_index > 0:
                        temp_history_index -= 1
                        current_text = history[temp_history_index]
                        cursor_pos = len(current_text)
                        refresh_line()
                        
                elif next2 == 'B':  # Down arrow
                    if history and temp_history_index < len(history) - 1:
                        temp_history_index += 1
                        current_text = history[temp_history_index]
                        cursor_pos = len(current_text)
                        refresh_line()
                    elif temp_history_index == len(history) - 1:
                        temp_history_index = len(history)
                        current_text = ""
                        cursor_pos = 0
                        refresh_line()
        
        elif char == '\x7f':  # Backspace
            if cursor_pos > 0:
                current_text = current_text[:cursor_pos-1] + current_text[cursor_pos:]
                cursor_pos -= 1
                refresh_line()
        
        elif char.isprintable():  # Regular character
            current_text = current_text[:cursor_pos] + char + current_text[cursor_pos:]
            cursor_pos += 1
            print(char, end='', flush=True)

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    speak(text)
    
    # Initialize history
    history = []
    history_index = 0
    
    # Add initial text to history if provided
    if len(sys.argv) > 1:
        history.append(text)
        history_index = len(history)
    
    print("Type text and press enter (Ctrl+C to exit).")
    print("Use ↑/↓ arrow keys to navigate through previous utterances.")
    
    try:
        while True:
            user_text, history_index = get_input_with_history("> ", history, history_index)
            if user_text.strip():
                speak(user_text)
                # Add to history, avoiding duplicates of the last entry
                if not history or history[-1] != user_text:
                    history.append(user_text)
                    # Keep only last 20 entries
                    if len(history) > 20:
                        history.pop(0)
                history_index = len(history)
    except KeyboardInterrupt:
        print("\nExiting...")
