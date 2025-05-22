import asyncio
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = None
model = None


def load_transcription_model():
    """Lazy load the transcription model to avoid heavy imports on module load."""
    global processor, model
    if processor is None or model is None:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model = model.to(dtype=torch.float64)


def process_audio(audio_data):
    """Transcribe audio data using Wav2Vec 2.0."""
    load_transcription_model()

    audio_input = np.squeeze(audio_data)

    if np.mean(np.abs(audio_input)) > 0.01:
        input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(dtype=torch.float64, device=model.device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription

    return "Audio is silent or near-silent."


async def process_audio_async(buffer):
    from speech_synthesis import speak
    import matplotlib.pyplot as plt
    import asyncio

    try:
        while True:
            if buffer:
                audio_data = buffer.pop(0)
                transcription = await asyncio.get_event_loop().run_in_executor(
                    None, process_audio, audio_data
                )
                speak(transcription.lower())

                plt.plot(audio_data)
                plt.title("Audio Waveform")
                plt.xlabel("TEXT: " + transcription[:70].lower() + "...")
                plt.ylabel("Amplitude")
                plt.savefig("waveform.png")
                plt.close()
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Error in process_audio_async: {str(e)}")
