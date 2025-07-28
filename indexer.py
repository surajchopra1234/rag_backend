# Import necessary libraries
from config import settings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
import chromadb

# Gemini client
gemini_client = genai.Client(api_key=settings.gemini_api_key)

# ChromaDB client
chromadb_client = chromadb.PersistentClient(path=settings.chroma_db_path)
collection = chromadb_client.get_or_create_collection(name="document_embeddings")

# ==================== Functions ====================>

def load_documents(file_path: str, file_extension: str):
    """
    Function to load a documents based on its file extension
    """

    if file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        # Raise an error for unsupported file types
        raise ValueError(f"Unsupported file type: {file_extension}")

    return loader.load()

def split_documents(documents):
    """
    Function to split documents into smaller chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def generate_embedding(content):
    """
    Function to generate embeddings for a content using Gemini API
    """

    result = gemini_client.models.embed_content(model="text-embedding-004", contents=content)
    return result.embeddings[0].values

def store_embeddings(split_docs, embeddings, file_name: str):
    """
    Function to store embeddings in ChromaDB collection
    """

    ids = [f"{file_name}_{i}" for i in range(len(split_docs))]
    documents = [doc.page_content for doc in split_docs]
    metadata = [{"file_name": file_name, "chunk_index": i} for i in range(len(split_docs))]

    # Add the embeddings to the collection
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata
    )

def add_document(file_path: str, file_name: str, file_extension: str):
    """
    Function to add a document to the knowledge base for RAG model
    """

    # Load the file
    docs = load_documents(file_path, file_extension)
    print(f"Number of documents loaded: {len(docs)}")

    # Split the documents into chunks
    split_docs = split_documents(docs)
    print(f"Number of documents after splitting: {len(split_docs)}")

    # Generate embeddings for each document
    embeddings = []

    for i, doc in enumerate(split_docs):
        print(f"Generating embeddings for document {i + 1}/{len(split_docs)}...")

        embedding = generate_embedding(doc.page_content)
        embeddings.append(embedding)

    # Store the embeddings
    store_embeddings(split_docs, embeddings, file_name)