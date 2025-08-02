# Import necessary libraries
from app.clients.gemini_client import gemini_client
# from sentence_transformers import SentenceTransformer

# Qwen embedding model (Local model)
# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def generate_embedding(content):
    """
    Function to generate embeddings for a content using Gemini API
    """

    # embeddings = model.encode(content)
    # return embeddings

    result = gemini_client.models.embed_content(model="gemini-embedding-001", contents=content)
    return result.embeddings[0].values
