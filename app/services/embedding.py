# Import necessary libraries
from app.clients.gemini_client import gemini_client

def generate_embedding(content):
    """
    Function to generate embeddings for a content using Gemini API
    """

    result = gemini_client.models.embed_content(model="gemini-embedding-001", contents=content)
    return result.embeddings[0].values
