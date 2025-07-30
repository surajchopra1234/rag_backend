# Import necessary libraries
import os
import json
from config import settings
from datetime import datetime
from typing import Generator
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.services.indexer import add_document, delete_document
from app.services.retriever import retrieve_and_generate
from app.services.speech import speech_to_text, text_to_speech

# Create FastAPI application instance
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for text queries
class TextQuery(BaseModel):
    query: str

class DeleteDocument(BaseModel):
    file_name: str


def stream_generator(query: str) -> Generator[str, None, None]:
    """
    Function to generate a stream of responses from the RAG model
    """

    try:
        for chunk in retrieve_and_generate(query):
            yield chunk.text
    except Exception as e:
        print(f"Error during response generation: {e}")
        yield "An error occurred while generating the response"


@app.get(
    "/api/documents",
    status_code=status.HTTP_200_OK
)
async def list_documents_endpoint():
    """
    API endpoint to list all documents in the knowledge base
    """

    try:
        # List all the files in the knowledge base
        os.makedirs(settings.data_directory, exist_ok=True)
        files = os.listdir(settings.data_directory)

        documents = []
        for file_name in files:
            if file_name.endswith('.json'):
                # Load metadata from the JSON file
                json_path = os.path.join(settings.data_directory, file_name)
                with open(json_path, 'r') as buffer:
                    metadata = json.load(buffer)
                    documents.append(metadata)

        return {
            'status': 'success',
            'message': 'Documents listed successfully',
            'documents': documents
        }

    except Exception as e:
        print(f"An error occurred while listing documents: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'An error occurred while listing documents: {str(e)}'
        )


@app.post(
    "/api/documents",
    status_code=status.HTTP_201_CREATED
)
async def upload_document_endpoint(file: UploadFile):
    """
    API endpoint to upload a document for indexing for RAG knowledge base
    """

    file_name = os.path.splitext(file.filename)[0]
    file_extension = os.path.splitext(file.filename)[1].lower()

    # Only TXT and PDF files are allowed
    allowed_extensions = {".txt", ".pdf"}
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Only PDF and TXT files are allowed"
        )

    try:
        # Read the file content
        content_bytes = await file.read()

        # Create the data directory if it does not exist
        os.makedirs(settings.data_directory, exist_ok=True)

        # Save the file to the data directory
        file_path = os.path.join(settings.data_directory, file.filename)
        with open(file_path, 'wb') as buffer:
            buffer.write(content_bytes)

        # Save the file metadata to a JSON file
        json_path = file_path + ".json"
        metadata = {
            "file_name": file_name,
            "file_extension": file_extension,
            "date": datetime.now().isoformat(),
            "size": len(content_bytes)
        }
        with open(json_path, 'w') as buffer:
            json.dump(metadata, buffer, indent=4)

        # Add the document to the knowledge base
        add_document(file_path, file_name, file_extension)

        return {
            'status': 'success',
            'message': 'Document uploaded successfully',
            'document': metadata
        }

    except Exception as e:
        print(f"An error occurred while uploading the document: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'An error occurred while uploading the document: {str(e)}'
        )


@app.delete(
    "/api/documents",
    status_code=status.HTTP_200_OK
)
async def delete_documents_endpoint(body: DeleteDocument):
    """
    API endpoint to delete a document from the knowledge base
    """

    try:
        # List all the files in the knowledge base
        files = os.listdir(settings.data_directory)

        for file_name in files:
            if file_name == body.file_name:
                # Delete the document file
                file_path = os.path.join(settings.data_directory, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)

                # Delete the document metadata JSON file
                json_path = file_path + ".json"
                if os.path.exists(json_path):
                    os.remove(json_path)

                # Remove the document from the knowledge base
                delete_document(body.file_name)

                break

        return {
            'status': 'success',
            'message': 'All documents deleted successfully'
        }

    except Exception as e:
        print(f"An error occurred while deleting documents: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'An error occurred while deleting documents: {str(e)}'
        )


@app.post(
    "/api/queries/text",
    status_code=status.HTTP_200_OK
)
async def queries_text_endpoint(body: TextQuery):
    """
    API endpoint to handle RAG queries in text format
    """
    print(f"Text query received: '{body.query}'")

    return StreamingResponse(
        stream_generator(body.query),
        media_type='text/plain'
    )


@app.post(
    "/api/queries/audio",
    status_code=status.HTTP_200_OK
)
async def queries_audio_endpoint(audio_file: UploadFile):
    """
    API endpoint to handle RAG queries in audio format
    """
    try:
        audio_bytes = await audio_file.read()
        query = speech_to_text(audio_bytes)
        print(f"Audio query received: '{query}'")

        if not query or query.isspace():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not transcribe any text from the provided audio file"
            )

        return StreamingResponse(
            stream_generator(query),
            media_type='text/plain'
        )
    except Exception as e:
        print(f"An error occurred in the audio query view: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the audio file"
        )


@app.post(
    "/api/text-to-speech",
    status_code=status.HTTP_200_OK
)
def text_to_speech_endpoint(body: TextQuery):
    """
    API endpoint to convert text to speech using Groq's TTS model
    """

    try:
        audio_bytes = text_to_speech(body.query)

        return Response(
            content=audio_bytes,
            media_type='audio/wav',
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )

    except Exception as e:
        print(f"An error occurred during text-to-speech conversion: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during text-to-speech conversion"
        )
