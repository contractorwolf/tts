"""Interactive script to cycle through TTS models and play a phrase."""

import curses
import torch
from TTS.api import TTS
import sounddevice as sd

# Full list of available speech models.  These mirror the commented list in
# ``speak.py``.  The arrow keys cycle through this list.
MODELS = [
    "tts_models/en/ek1/tacotron2",
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/tacotron2-DDC_ph",
    "tts_models/en/ljspeech/glow-tts",
    "tts_models/en/ljspeech/speedy-speech",
    "tts_models/en/ljspeech/tacotron2-DCA",
    "tts_models/en/ljspeech/vits",
    "tts_models/en/ljspeech/vits--neon",
    "tts_models/en/ljspeech/fast_pitch",
    "tts_models/en/ljspeech/overflow",
    "tts_models/en/ljspeech/neural_hmm",
    "tts_models/en/vctk/vits",
    "tts_models/en/vctk/fast_pitch",
    "tts_models/en/sam/tacotron-DDC",
    "tts_models/en/blizzard2013/capacitron-t2-c50",
    "tts_models/en/blizzard2013/capacitron-t2-c150_v2",
    "tts_models/en/multi-dataset/tortoise-v2",
    "tts_models/en/jenny/jenny",
]

# Determine whether to use the GPU or CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Text that will be spoken whenever a new model is selected.
DEFAULT_TEXT = "the journey of a thousand miles begins with a single step"


# Cache for loaded models so we don't reload the same model twice.
_loaded_models = {}

# Start with the speedy-speech model as it loads quickly.
_current_index = MODELS.index("tts_models/en/ljspeech/speedy-speech")


def _get_tts(idx: int) -> TTS:
    """Return a cached ``TTS`` instance for the model at ``idx``."""

    name = MODELS[idx]
    if name not in _loaded_models:
        print(f"Loading model: {name}")
        _loaded_models[name] = TTS(model_name=name).to(DEVICE)
    else:
        print(f"Using cached model: {name}")
    return _loaded_models[name]


def _speak(tts: TTS, text: str = DEFAULT_TEXT) -> None:
    """Convert ``text`` to speech using ``tts`` and play it."""

    print(f"Speaking with {MODELS[_current_index]}: '{text}'")
    wav = tts.tts(text=text)
    sd.play(wav, samplerate=22050)
    sd.wait()


def _update_display(stdscr: "curses._CursesWindow") -> None:
    """Refresh the screen with the current model and key hints."""

    stdscr.clear()
    stdscr.addstr(0, 0, f"Current model: {MODELS[_current_index]}")
    stdscr.addstr(2, 0, "UP/DOWN: change model & speak | SPACE: show model | q: quit")
    stdscr.refresh()


def main(stdscr: "curses._CursesWindow") -> None:
    """Entry point for ``curses.wrapper``. Handles key presses."""

    global _current_index
    curses.curs_set(0)
    stdscr.nodelay(True)
    tts = _get_tts(_current_index)
    _speak(tts)
    _update_display(stdscr)

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP:
            _current_index = (_current_index - 1) % len(MODELS)
            print(f"Switched to: {MODELS[_current_index]}")
            tts = _get_tts(_current_index)
            _speak(tts)
            _update_display(stdscr)
        elif key == curses.KEY_DOWN:
            _current_index = (_current_index + 1) % len(MODELS)
            print(f"Switched to: {MODELS[_current_index]}")
            tts = _get_tts(_current_index)
            _speak(tts)
            _update_display(stdscr)
        elif key == ord(" "):
            print(f"Current model is: {MODELS[_current_index]}")
            _update_display(stdscr)
        elif key == ord("q"):
            print("Exiting...")
            break
        curses.napms(100)


if __name__ == "__main__":
    print("Starting model chooser. Press 'q' to quit.")
    curses.wrapper(main)
