import asyncio

from recording import record_audio_async
from transcription import process_audio_async


async def main():
    buffer = []
    print("----------------------------------------------------")
    print("Starting Listening for Audio...")
    print("----------------------------------------------------")

    await asyncio.gather(
        record_audio_async(buffer),
        process_audio_async(buffer),
    )


if __name__ == "__main__":
    asyncio.run(main())
