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


def retrieve_and_generate(query: str, short: bool):
    """
    Function to retrieve documents based on a query and generate a response using the Gemini model
    """

    # Define the system instruction for the Gemini model
    system_instruction = """ You are an AI assistant designed to answer questions based on a given set of documents.
        - **Primary Goal**: Your first priority is to answer questions using only the information found in the Retrieved Documents. Synthesize the relevant information into a clear and concise answer. Do not invent facts or assume information not present in the text.
        - **Synthesize, Don't Just Extract**: Combine information from different parts of the documents to create a comprehensive, well-written answer. Avoid simply copying and pasting chunks of text.
        - **Handle Missing Information**: If the documents do not contain the necessary information to fully answer the question, you must state this clearly at the beginning of your response. For example, say: "The provided documents do not contain information on this topic."
        - **Fallback to General Knowledge (Conditional)**: This is a critical step. Only after you have explicitly stated that the documents lack the necessary information may you provide an answer from your own knowledge base.
             - Clear Separation: You must create a clear distinction between the document-based analysis and your general knowledge.
             - Introduce Your Knowledge: Use a clear introductory phrase. For instance:"However, based on my general knowledge..."
             - Purpose: This two-step process ensures transparency and builds user trust by making it absolutely clear what information is verified by the documents and what is not.
    """

    # Add short response instructions if the short flag is set
    if short:
        system_instruction += """
             - **IMPORTANT - Keep responses very short and concise**: Your responses will be used for text-to-audio conversion. 
               Provide brief, conversational answers that would sound natural when spoken. 
               Limit responses to 1 short paragraphs maximum.
               Use simple language and shorter sentences.
        """

    # Retrieve documents based on the query
    documents = retrieve_documents(query)

    # Generate a response using the Gemini model
    response = gemini_client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=[
            f"Question: {query}.",
            f"Retrieved Documents: {documents}.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_budget=10)
        )
    )

    return response
