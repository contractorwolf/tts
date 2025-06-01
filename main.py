# sduo apt install -y portaudio19-dev


import torch
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import asyncio 
import pyaudio
import soundfile as sf
import matplotlib.pyplot as plt


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC



from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset
from transformers import SpeechT5HifiGan
from pydub import AudioSegment
from pydub.playback import play

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

speech_model = "tts_models/en/ljspeech/speedy-speech"  # Change this to the desired model

tts = TTS(model_name=speech_model).to(device)

processor2 = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

model2 = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

output_path = "outputs/audio.wav"
# text = "the shieks sixth sheep is sick"
# text = "round and round the rugged rock, the ragged rascal ran"
# text = "Ok, so you do want a loan, but you don't think you can get approved with your credit score. Do I have that right?"
text = "ok i am listening, please speak clearly and i will transcribe your speech into text. Then i will speak your words back to you. Whenever you are ready..."

wav = tts.tts(text=text)
# Play the audio (sample rate is typically 22050 Hz for most TTS models)
sd.play(wav, samplerate=22050)
sd.wait()  # Wait until the audio is finished playing

# tts.tts_to_file(text=text, file_path=output_path)

def record_audio():
    """
    Records audio until 2 seconds of silence are detected, then returns the recorded audio as a NumPy array.
    
    Returns:
        np.array: Recorded audio data.
    """
    
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 1              # Number of audio channels (1 for mono, 2 for stereo)
    RATE = 16000              # Sampling rate in Hz
    CHUNK = 1024              # Number of frames per buffer
    SILENCE_LIMIT = 1.5         # Silence limit in seconds
    SILENCE_FRAME_LIMIT = (RATE // CHUNK) * SILENCE_LIMIT  # Number of silence frames to stop recording
    THRESHOLD = 25
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    silent_frames = 0
    
    # print("Recording...")
    
    while True:
        # Record audio
        data = stream.read(CHUNK)
        current_frame = np.frombuffer(data, dtype=np.int16)
        frames.append(data)
        
        # Check for silence
        #rms = np.sqrt(np.mean(current_frame**2))
        rms = np.sqrt(np.abs(np.mean(current_frame**2)))
        
        if rms < THRESHOLD:  # Assuming 20 is the silence threshold
            silent_frames += 1
        else:
            print ("Sound Level:", rms)
            silent_frames = 0
        
        # If silence for 2 seconds, stop recording
        if silent_frames >= SILENCE_FRAME_LIMIT:
            break
    
    #print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    
    # Calculate RMS
    rms = np.sqrt(np.mean(audio_data**2))
    
    
    # Only return audio data if it exceeds a certain RMS threshold
    if rms > THRESHOLD:
        print ("Sound Level rms:", rms)
        return audio_data
    else:
        return None



# Load pre-trained model and processor from Hugging Face Model Hub
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Move the entire model to float64 dtype
model = model.to(dtype=torch.float64)


def speak(text):
    print("Converting text to speech...")   
    print("----------------------------------------------------")        
    
    inputs = processor2(text=text, return_tensors="pt")
    speech = model2.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Convert the NumPy array to audio
    audio = AudioSegment(
        speech.numpy().tobytes(),
        frame_rate=16000,
        sample_width=speech.numpy().dtype.itemsize,
        channels=1
    )
    
    print("Speaking...")   

    # Play the audio
    play(audio)




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
        input_values = input_values.to(dtype=torch.float64, device=model.device)
        
        # Perform inference with the model
        with torch.no_grad():
            try:
                logits = model(input_values).logits
                # print(f"Input values shape: {input_values.shape}, values: {input_values}")
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
    try:
        while True:
            # audio_data = record_audio()
            # print("Recording...")
            # buffer.append(audio_data)
            # print("Buffer length:", len(buffer))
            # await asyncio.sleep(0.1)  # adjust accordingly
            
            audio_data = record_audio()
            if audio_data is not None:
                buffer.append(audio_data)
            await asyncio.sleep(0.1)  # adjust accordingly
    
    except Exception as e:
        print(f"Error in record_audio_async: {str(e)}")
        
        
        
        

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
