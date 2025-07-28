# Import necessary libraries
from config import settings
from google import genai
from google.genai import types
import chromadb

# Gemini client
gemini_client = genai.Client(api_key=settings.gemini_api_key)

# ChromaDB client
chromadb_client = chromadb.PersistentClient(path=settings.chroma_db_path)
collection = chromadb_client.get_collection(name="document_embeddings")

# Function to generate embeddings
def generate_embeddings(document):
    """
    Function to generate embeddings for a given document using the Gemini API
    """
    result = gemini_client.models.embed_content(model="text-embedding-004", contents=document)
    return result.embeddings[0].values

# Function to retrieve documents
def retrieve_documents(query: str):
    """
    Function to retrieve documents based on a query
    """
    query_embeddings = generate_embeddings(query)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=3,
        include=["documents", "metadatas"]
    )

    return results['documents'][0]

# Function to retrieve and generate a response
def retrieve_and_generate(query: str):
    """
    Function to retrieve documents based on a query and generate a response using the Gemini model
    """
    # Retrieve documents based on the query
    documents = retrieve_documents(query)

    # Generate a response using the Gemini model
    response = gemini_client.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=[
            f"Question: {query}.",
            f"Retrieved Documents: {documents}.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=""" You are a knowledgeable assistant that answers questions based on retrieved documents.
                - Only use information from the provided documents to answer.
                - If the documents don't contain relevant information, acknowledge this limitation.
                - Format your answers clearly and concisely. """,
            thinking_config=types.ThinkingConfig(thinking_budget=512)
        )
    )

    return response

# Function to convert the speech to the text
def speech_to_text(audio_bytes: bytes, content_type: str):
    """
    Transcribes an audio file to text using the Gemini API
    """
    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            'Please transcribe the following audio file.',
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=content_type,
            )
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=2)
        )
    )

    # Return the transcribed text
    return response.text if hasattr(response, 'text') else ""