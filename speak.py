# for: ljspeech/vits
# needed: sudo apt-get install espeak espeak-data
# needed: pip install espeak

# needed: pip install termios
# needed: pip install tty
# needed: pip install select

# TODO: Future improvements for speed and functionality
# - [ ] Implement async audio generation to overlap TTS processing with playback
# - [ ] Add voice cloning support for custom speaker embeddings
# - [ ] Implement streaming TTS for real-time generation of long texts
# - [ ] Add SSML support for better prosody control (emphasis, pauses, speed)
# - [ ] Cache audio to disk for persistence across sessions
# - [ ] Add hotkey support (e.g., spacebar to interrupt/skip current speech)
# - [ ] Implement batch processing for multiple texts at once
# - [ ] Add voice selection menu for multi-speaker models
# - [ ] Optimize model loading with quantization for faster startup
# - [ ] Add text preprocessing (abbreviation expansion, number normalization)
# - [ ] Implement audio effects (speed control, pitch adjustment)
# - [ ] Add export functionality to save generated audio as files

import sys
import torch
from TTS.api import TTS  # Text-to-speech library
import sounddevice as sd  # Audio playback library
import termios  # Terminal I/O control
import tty  # Terminal control functions
import re  # Add this import at the top with other imports
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU
DEFAULT_TEXT = "i am ready to speak what you type on the command line. click the arrow up or down to go to the previous statements."  # Default text to speak
# Minimum number of characters for each utterance.  Shorter phrases cause
# errors in some models, so they are padded with spaces up to this length.
MIN_INPUT_LEN = 20  # Minimum text length to avoid model errors

# Add this near the top with other globals
AUDIO_CACHE = {}  # Cache for storing generated audio to avoid regenerating same text
MAX_CACHE_SIZE = 100  # Limit cache size to prevent memory issues

print("Loading TTS model...")
# load the speedy-speech model once
tts = TTS(model_name=speech_model).to(DEVICE)  # Initialize TTS model and move to GPU/CPU

def speak_streaming(text: str) -> None:
    """Convert text to speech sentence by sentence for faster initial playback."""
    text = text.strip()  # Remove extra whitespace
    if not text:  # Skip if empty
        print("DEBUG: Text is empty, returning")  # Debug print
        return
    
    # Split text into sentences using regex
    sentences = re.split(r'[.!?]+', text)  # Split on sentence endings
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences

    for i, sentence in enumerate(sentences):  # Process each sentence separately
        print(f"DEBUG: Processing sentence {i+1}/{len(sentences)}: '{sentence}'")  # Debug print
        #prepared_sentence = _prepare_text(sentence)  # Ensure minimum length
        print(f"Speaking: {sentence}")  # Show current sentence
        
        # Check cache first
        if sentence in AUDIO_CACHE:  # Use cached audio if available
            print("(using cached audio)")
            wav = AUDIO_CACHE[sentence]
        else:
            print("DEBUG: Generating new audio...")  # Debug print
            wav = tts.tts(text=sentence)  # Generate audio for this sentence
            print(f"DEBUG: Generated audio with shape: {len(wav) if hasattr(wav, '__len__') else 'unknown'}")  # Debug print
            
            # Manage cache size
            if len(AUDIO_CACHE) >= MAX_CACHE_SIZE:  # Remove oldest if cache full
                oldest_key = next(iter(AUDIO_CACHE))
                del AUDIO_CACHE[oldest_key]
            
            AUDIO_CACHE[sentence] = wav  # Cache the new audio
            print(f"(cached - total cached: {len(AUDIO_CACHE)})")
        
        print("DEBUG: Starting playback...")  # Debug print
        sd.play(wav, samplerate=22050)  # Play this sentence immediately
        sd.wait()  # Wait for sentence to finish before next one
        print("DEBUG: Playback completed")  # Debug print

def speak(text: str) -> None:
    """Convert text to speech - uses streaming for long texts."""
    print(f"DEBUG: speak() called with text: '{text}' (length: {len(text)})")  # Debug print
    text = text.strip()  # Clean input text
    
    # Use streaming for longer texts (more than 100 characters)
    if len(text) > 100:  # Arbitrary threshold for "long" text
        print("DEBUG: Using streaming mode (text > 100 chars)")  # Debug print
        speak_streaming(text)  # Use sentence-by-sentence playback
    else:
        # Keep original behavior for short texts
        print(f"Speaking: {text}")
        
        if text in AUDIO_CACHE:  # Check cache
            print("(using cached audio)")
            wav = AUDIO_CACHE[text]
        else:
            print("DEBUG: Generating new audio...")  # Debug print
            wav = tts.tts(text=text)  # Generate audio
            print(f"DEBUG: Generated audio with shape: {len(wav) if hasattr(wav, '__len__') else 'unknown'}")  # Debug print
            
            if len(AUDIO_CACHE) >= MAX_CACHE_SIZE:  # Manage cache
                oldest_key = next(iter(AUDIO_CACHE))
                del AUDIO_CACHE[oldest_key]
            
            AUDIO_CACHE[text] = wav  # Cache audio
            print(f"(cached - total cached: {len(AUDIO_CACHE)})")
        
        print("DEBUG: Starting playback...")  # Debug print
        sd.play(wav, samplerate=22050)  # Play audio
        sd.wait()  # Wait for completion
        print("DEBUG: Playback completed")  # Debug print


def get_char():
    """Get a single character from stdin without pressing enter"""
    fd = sys.stdin.fileno()  # Get file descriptor for stdin
    old_settings = termios.tcgetattr(fd)  # Save current terminal settings
    try:
        tty.setraw(sys.stdin.fileno())  # Set terminal to raw mode (no buffering)
        ch = sys.stdin.read(1)  # Read single character
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # Restore terminal settings
    return ch

def get_input_with_history(prompt, history, history_index):
    """Get input with arrow key history navigation"""
    print(prompt, end='', flush=True)  # Display prompt without newline
    current_text = ""  # Current text being typed
    cursor_pos = 0  # Current cursor position (unused in this implementation)
    temp_history_index = history_index  # Temporary index for navigating history
    
    def refresh_line():
        """Clear current line and redraw prompt + current text"""
        print(f'\r\033[K{prompt}{current_text}', end='', flush=True)  # Clear line and redraw
    
    while True:
        char = get_char()  # Get single character input
        
        if char == '\r' or char == '\n':  # Enter key pressed
            print()  # Move to next line
            return current_text, len(history)  # Return text and reset history index
        
        elif char == '\x03':  # Ctrl+C pressed
            raise KeyboardInterrupt  # Exit program
        
        elif char == '\x1b':  # Escape sequence (arrow keys)
            next1, next2 = get_char(), get_char()  # Read next two characters
            if next1 == '[':  # ANSI escape sequence
                if next2 == 'A':  # Up arrow
                    if history and temp_history_index > 0:  # If history exists and not at beginning
                        temp_history_index -= 1  # Move back in history
                        current_text = history[temp_history_index]  # Load previous text
                        cursor_pos = len(current_text)  # Move cursor to end
                        refresh_line()  # Redraw line
                        
                elif next2 == 'B':  # Down arrow
                    if history and temp_history_index < len(history) - 1:  # If not at end of history
                        temp_history_index += 1  # Move forward in history
                        current_text = history[temp_history_index]  # Load next text
                        cursor_pos = len(current_text)  # Move cursor to end
                        refresh_line()  # Redraw line
                    elif temp_history_index == len(history) - 1:  # If at last history item
                        temp_history_index = len(history)  # Move past history
                        current_text = ""  # Clear current text
                        cursor_pos = 0  # Reset cursor
                        refresh_line()  # Redraw line
        
        elif char == '\x7f':  # Backspace key
            if cursor_pos > 0:  # If there's text to delete
                current_text = current_text[:cursor_pos-1] + current_text[cursor_pos:]  # Remove character
                cursor_pos -= 1  # Move cursor back
                refresh_line()  # Redraw line
        
        elif char.isprintable():  # Regular printable character
            current_text = current_text[:cursor_pos] + char + current_text[cursor_pos:]  # Insert character
            cursor_pos += 1  # Move cursor forward
            print(char, end='', flush=True)  # Display character

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT  # Use command line arg or default
    speak(text)  # Speak the initial text
    
    # Initialize history
    history = []  # List to store previous utterances
    history_index = 0  # Current position in history
    
    # Add initial text to history if provided
    if len(sys.argv) > 1:  # If command line argument was provided
        history.append(text)  # Add it to history
        history_index = len(history)  # Set index to end of history
    
    print("Type text and press enter (Ctrl+C to exit).")
    print("Use ↑/↓ arrow keys to navigate through previous utterances.")
    
    try:
        while True:  # Main input loop
            user_text, history_index = get_input_with_history("> ", history, history_index)  # Get user input
            if user_text.strip():  # If user entered non-empty text
                speak(user_text)  # Convert to speech and play
                # Add to history, avoiding duplicates of the last entry
                if not history or history[-1] != user_text:  # If history empty or text is different
                    history.append(user_text)  # Add to history
                    # Keep only last 20 entries
                    if len(history) > 20:  # If history too long
                        history.pop(0)  # Remove oldest entry
                history_index = len(history)  # Reset history index to end
    except KeyboardInterrupt:  # Ctrl+C pressed
        print("\nExiting...")  # Clean exit message
