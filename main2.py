# sduo apt install -y portaudio19-dev


import torch
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import asyncio 
import pyaudio
import matplotlib.pyplot as plt
import re  # Add this import at the top with other imports


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# import soundfile as sf
# from pydub import AudioSegment
# from pydub.playback import play




device = "cuda" if torch.cuda.is_available() else "cpu"



# 10: tts_models/en/ek1/tacotron2
# 11: tts_models/en/ljspeech/tacotron2-DDC
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

speech_model = "tts_models/en/ljspeech/vits"  # Change this to the desired model

tts = TTS(model_name=speech_model).to(device)


# instructions
text = "ok i am listening, please speak clearly and i will speak your words back to you. Whenever you are ready..."

wav = tts.tts(text=text)
# Play the audio (sample rate is typically 22050 Hz for most TTS models)
sd.play(wav, samplerate=22050)
sd.wait()  # Wait until the audio is finished playing


# Audio recording configuration
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1              # Number of audio channels (mono)
RATE = 16000              # Sampling rate in Hz
CHUNK = 1024              # Frames per buffer
SILENCE_LIMIT = 1.5       # Silence limit in seconds
SILENCE_FRAME_LIMIT = int((RATE // CHUNK) * SILENCE_LIMIT)
THRESHOLD = 30




# for: ljspeech/vits
# needed: sudo apt-get install espeak espeak-data
# needed: pip install espeak

# needed: pip install termios
# needed: pip install tty
# needed: pip install select

print("Loading TTS model...")





#speech_model = "tts_models/en/ljspeech/speedy-speech" #"tts_models/en/ljspeech/speedy-speech"
speech_model = "tts_models/en/ljspeech/vits"  # Changed from speedy-speech to vits

# determine if GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU
DEFAULT_TEXT = "i am ready to speak what you type on the command line. click the arrow up or down to go to the previous statements."  # Default text to speak


# Add this near the top with other globals
AUDIO_CACHE = {}  # Cache for storing generated audio to avoid regenerating same text
MAX_CACHE_SIZE = 50  # Limit cache size to prevent memory issues

LISTENING = True  # Global flag to control listening state

print("Loading TTS model...")
# load the speedy-speech model once
tts = TTS(model_name=speech_model).to(DEVICE)  # Initialize TTS model and move to GPU/CPU

# Load pre-trained model and processor from Hugging Face Model Hub
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Use float32 precision and move the STT model to the selected device
model = model.to(device=device, dtype=torch.float32)


def speak_streaming(text: str) -> None:
    """Convert text to speech sentence by sentence for faster initial playback."""
    global LISTENING  # Use global flag to control listening state
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
        LISTENING = False  # Set listening state to False before playback
        sd.play(wav, samplerate=22050)  # Play this sentence immediately
        sd.wait()  # Wait for sentence to finish before next one
        LISTENING = True  # Set listening state back to True after playback
        print("DEBUG: Playback completed")  # Debug print

def speak(text: str) -> None:
    """Convert text to speech - uses streaming for long texts."""
    global LISTENING  # Use global flag to control listening state


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
        LISTENING = False  # Set listening state to False before playback
        sd.play(wav, samplerate=22050)  # Play this sentence immediately
        sd.wait()  # Wait for sentence to finish before next one
        LISTENING = True  # Set listening state back to True after playback
        print("DEBUG: Playback completed")  # Debug print




def record_audio(stream):
    """Record audio from an open stream until silence is detected.

    Args:
        stream: An open ``pyaudio`` input stream.

    Returns:
        np.array or None: The recorded audio data or ``None`` if below ``THRESHOLD``.
    """
    
    frames = []
    silent_frames = 0
    
    # print("Recording...")
    
    while True:
        # Record audio

        if not LISTENING:  # Check if we are currently listening
            print("Not listening, skipping recording...")
            continue
        # print("Listening...")
        data = stream.read(CHUNK)
        current_frame = np.frombuffer(data, dtype=np.int16)
        frames.append(data)
        
        # Check for silence
        #rms = np.sqrt(np.mean(current_frame**2))
        rms = np.sqrt(np.abs(np.mean(current_frame**2)))
        
        if rms < THRESHOLD:  # Assuming 20 is the silence threshold
            # print("Silence detected, RMS:", rms)
            # print("Silent Frames:", silent_frames)
            silent_frames += 1
        else:
            print ("Sound Level:", rms)
            silent_frames = 0
        
        # If silence for 2 seconds, stop recording
        if silent_frames >= SILENCE_FRAME_LIMIT:
            break
    
    #print("Finished recording.")
    
    
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    
    # Calculate RMS
    rms = np.sqrt(np.mean(audio_data**2))
    
    
    # Only return audio data if it exceeds a certain RMS threshold
    if rms > THRESHOLD:
        print ("Sound Level rms:", rms)
        return audio_data
    else:
        return None










def process_audio(audio_data):
    """
    Transcribes audio data using the Wav2Vec 2.0 model from the transformers library.
    
    Args:
        audio_data (np.array): The audio data to transcribe.
    
    Returns:
        str: The transcription of the audio data.
    """
    
    # Ensure the audio is a 1-D numpy array
    audio_input = np.squeeze(audio_data)
    
    # Check if the audio input is not silent or near-silent
    if np.mean(np.abs(audio_input)) > 0.01:
        
        # Preprocess the audio data and prepare the input features
        input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values
        
        # Convert input_values to the appropriate data type and device
        input_values = input_values.to(dtype=torch.float32, device=device)
        
        # Perform inference with the model
        with torch.no_grad():
            try:
                logits = model(input_values).logits
                # print(f"Logits shape: {logits.shape}, values: {logits}")
            except Exception as e:
                print(f"Error during model inference: {str(e)}")
                print(f"Input values dtype: {input_values.dtype}, device: {input_values.device}")
                return "Error during transcription"
        
        # Identify the predicted tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode the ids to text
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription
    
    else:
        return "Audio is silent or near-silent."



async def record_audio_async(buffer):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    try:
        while True:
            audio_data = record_audio(stream)
            if audio_data is not None:
                buffer.append(audio_data)
            await asyncio.sleep(0.1)  # adjust accordingly

    except Exception as e:
        print(f"Error in record_audio_async: {str(e)}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        
        
        

async def process_audio_async(buffer):
    try:
        while True:
            # print("Checking buffer...")
            if buffer:
                print("----------------------------------------------------")
                print("Processing audio...")
                audio_data = buffer.pop(0)
                
                # Instead of using a ThreadPoolExecutor inside an async function, you might consider using run_in_executor with None as the executor to use the default ThreadPoolExecutor:                
                # Use a thread pool for blocking/synchronous code
                # with concurrent.futures.ThreadPoolExecutor() as pool:
                    #transcription = await asyncio.get_event_loop().run_in_executor(pool, process_audio, audio_data)

                transcription = await asyncio.get_event_loop().run_in_executor(None, process_audio, audio_data)
                print("----------------------------------------------------")
                print("Transcription:", transcription.lower())
                
                print("----------------------------------------------------")
                speak(transcription.lower())
                
                print("----------------------------------------------------")
                print("Listening...")   
                print("----------------------------------------------------")
                
                print("begin post process")
                # Visualize the audio waveform
                plt.plot(audio_data)
                plt.title('Audio Waveform')
                plt.xlabel('TEXT: ' + transcription[0:70].lower() + '...')
                plt.ylabel('Amplitude')
                # plt.show()
                # Save the plot as a PNG file
                plt.savefig("waveform.png")
                plt.close()  # Close the figure window             
                
                # # Save a short snippet of audio for debugging
                # sf.write('debug_audio.wav', audio_data, 16000)       
                # print("end post process")
            # else:
                # print("Waiting for audio data...")
            await asyncio.sleep(0.1)  # adjust accordingly
    except Exception as e:
        print(f"Error in process_audio_async: {str(e)}")


async def main():
    buffer = []
    print("----------------------------------------------------")
    print("Starting Listening for Audio...")
    print("----------------------------------------------------")
    
    await asyncio.gather(
        record_audio_async(buffer),
        process_audio_async(buffer)
    )

asyncio.run(main())
