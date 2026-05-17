import os
import uuid
import shutil
import chromadb
from pypdf import PdfReader
from chromadb.utils import embedding_functions


DOCS_FOLDER = "docs"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "my_documents"


def reset_chroma_db():
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print("Old ChromaDB deleted.")


def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def load_documents(folder_path):
    documents = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if file_name.lower().endswith(".txt"):
            text = read_txt_file(file_path)

            documents.append({
                "file_name": file_name,
                "text": text
            })

        elif file_name.lower().endswith(".pdf"):
            text = read_pdf_file(file_path)

            documents.append({
                "file_name": file_name,
                "text": text
            })

    return documents


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


def create_chroma_collection():
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return collection


def ingest_documents():
    reset_chroma_db()

    print("Loading documents...")

    documents = load_documents(DOCS_FOLDER)

    if not documents:
        print("No documents found in docs folder.")
        return

    collection = create_chroma_collection()

    ids = []
    texts = []
    metadatas = []

    for doc in documents:
        file_name = doc["file_name"]
        text = doc["text"]

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            texts.append(chunk)
            metadatas.append({
                "source": file_name,
                "chunk_index": index
            })

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Successfully stored {len(texts)} chunks from {len(documents)} documents in ChromaDB.")


if __name__ == "__main__":
    ingest_documents()