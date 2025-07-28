# Import necessary libraries
from typing import Generator
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from indexer import index_documents
from retriever import retrieve_and_generate, speech_to_text

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

@app.post(
    "/api/index/documents",
    status_code=status.HTTP_200_OK
)
def index_documents_endpoint():
    """
    API endpoint to trigger the document indexing process
    """
    try:
        print("Indexing of the documents has started")
        index_documents()
        print("Indexing of the documents has finished")

        return {
            'status': 'success',
            'message': 'Documents indexing process finished successfully'
        }
    except Exception as e:
        print(f"An error occurred during indexing: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'An error occurred during indexing: {str(e)}'
        )

@app.post(
    "/api/queries/text",
    status_code=status.HTTP_200_OK
)
async def queries_text_endpoint(item: TextQuery):
    """
    API endpoint to handle RAG queries in text format
    """
    print(f"Text query received: '{item.query}'")

    return StreamingResponse(
        stream_generator(item.query),
        media_type='text/plain'
    )

@app.post(
    "/api/queries/audio",
    status_code=status.HTTP_200_OK
)
async def queries_audio_endpoint(audio_file: UploadFile = File(...)):
    """
    API endpoint to handle RAG queries in audio format
    """
    try:
        audio_bytes = await audio_file.read()
        query = speech_to_text(audio_bytes, audio_file.content_type)
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