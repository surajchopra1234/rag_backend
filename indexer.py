# Import necessary libraries
from config import settings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
import chromadb

# Gemini client
gemini_client = genai.Client(api_key=settings.gemini_api_key)

# ChromaDB client
chromadb_client = chromadb.PersistentClient(path=settings.chroma_db_path)
collection = chromadb_client.get_or_create_collection(name="document_embeddings")

# Function to load the file
def load_file(file_path):
    """
    Function to load a text file and return its documents
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents

# Function to split documents
def split_documents(documents):
    """
    Function to split documents into smaller chunks for better processing
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Function to generate embeddings
def generate_embeddings(document):
    """
    Function to generate embeddings for a given document using Gemini API
    """
    result = gemini_client.models.embed_content(model="text-embedding-004", contents=document)
    return result.embeddings[0].values

# Function to store embeddings
def store_embeddings(documents, embeddings_list):
    """
    Function to store embeddings in ChromaDB collection
    """
    ids = [str(i) for i in range(len(documents))]
    documents_content = [doc.page_content for doc in documents]
    metadata = [doc.metadata for doc in documents]

    # Add the embeddings to the collection
    collection.upsert(
        ids=ids,
        documents=documents_content,
        embeddings=embeddings_list,
        metadatas=metadata
    )

# Function to index documents
def index_documents():
    """
    Function to load data from a file, generate embeddings, and store them in ChromaDB.
    """
    # Load the file
    docs = load_file(settings.data_file_path)
    print(f"Number of documents loaded: {len(docs)}")

    # Split the documents
    split_docs = split_documents(docs)
    print(f"Number of documents after splitting: {len(split_docs)}")

    # Generate embeddings for each document
    embeddings_list = []

    for i, doc in enumerate(split_docs):
        print(f"Generating embeddings for document {i + 1}/{len(split_docs)}...")

        embeddings = generate_embeddings(doc.page_content)
        embeddings_list.append(embeddings)

    # Store the embeddings in Chroma
    store_embeddings(split_docs, embeddings_list)