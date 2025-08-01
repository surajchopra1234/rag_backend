# Import necessary libraries
from sentence_transformers import SentenceTransformer

# Qwen embedding model
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def generate_embedding(content):
    """
    Function to generate embeddings for a content using Gemini API
    """

    embeddings = model.encode(content)
    return embeddings
