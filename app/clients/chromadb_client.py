# Import necessary libraries
from config import settings
import chromadb

# Create a ChromaDB client
chromadb_client = chromadb.PersistentClient(path=settings.chroma_db_path)

# Get or create a collection for document embeddings
collection = chromadb_client.get_or_create_collection(name="document_embeddings")
