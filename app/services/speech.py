# Import necessary libraries
from app.clients.groq_client import groq_client
from fastapi import HTTPException


def speech_to_text(audio_bytes: bytes) -> str:
    """
    Function to convert speech to text using Groq's Whisper model
    """

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model="whisper-large-v3-turbo",
            prompt="Please transcribe the provided audio file. The speaker has an Indian English accent.",
            response_format="verbose_json",
            language="en"
        )

        return transcription.text

    except Exception as e:
        print(f"Error during speech-to-text conversion: {e}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred during speech-to-text conversion"
        )
