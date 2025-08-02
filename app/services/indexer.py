# Import necessary libraries
from app.services.embedding import generate_embedding
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.clients.chromadb_client import collection


def load_file(file_path: str, file_extension: str):
    """
    Function to load a file based on its extension (.txt or .pdf)
    """

    if file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        # Raise an error for unsupported file types
        raise ValueError(f"Unsupported file type: {file_extension}")

    return loader.load()


def split_documents(documents, chunk_size, chunk_overlap):
    """
    Function to split documents into smaller chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def store_embeddings(split_docs, embeddings, file_name: str):
    """
    Function to store embeddings in the ChromaDB collection
    """

    ids = [f"{file_name}_{i}" for i in range(len(split_docs))]
    documents = [doc.page_content for doc in split_docs]
    metadata = [{"file_name": f"{file_name}", "chunk_index": i} for i in range(len(split_docs))]

    # Add the documents and embeddings to the collection
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata
    )


def add_document(file_path: str, file_name: str, file_extension: str, chunk_size=1000, chunk_overlap=200, sleep_time=0):
    """
    Function to add a document to the sources of the RAG model
    """

    # Load the file
    docs = load_file(file_path, file_extension)

    # Split the documents into chunks
    split_docs = split_documents(docs, chunk_size, chunk_overlap)
    print(f"Number of split documents: {len(split_docs)}")

    # Generate embeddings for each document
    embeddings = []

    for index, doc in enumerate(split_docs):
        print(f"Generating embeddings for document {index + 1}/{len(split_docs)}...")

        embedding = generate_embedding(doc.page_content)
        embeddings.append(embedding)

    # Store the embeddings
    store_embeddings(split_docs, embeddings, f"{file_name}{file_extension}")


def delete_document(file_name: str):
    """
    Function to delete a document from the sources of the RAG model
    """

    # Delete the document from the collection
    collection.delete(where={"file_name": file_name})
    print(f"Document {file_name} deleted successfully")
