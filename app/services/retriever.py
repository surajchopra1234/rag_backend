# Import necessary libraries
from google.genai import types
from app.clients.gemini_client import gemini_client
from app.services.embedding import generate_embedding
from app.clients.chromadb_client import collection

def retrieve_documents(query: str):
    """
    Function to retrieve documents based on a query
    """

    query_embeddings = generate_embedding(query)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=3,
        include=["documents", "metadatas"]
    )

    return results['documents'][0]


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
