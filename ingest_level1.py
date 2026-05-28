import os
import uuid
import shutil
import chromadb

from pypdf import PdfReader
from chromadb.utils import embedding_functions

# ==========================================
# CONFIGURATION
# ==========================================

DOCS_FOLDER = "docs"

CHROMA_DB_PATH = "chroma_db"

COLLECTION_NAME = "my_documents"

# ==========================================
# RESET CHROMADB
# ==========================================

def reset_chroma_db():

    if os.path.exists(CHROMA_DB_PATH):

        shutil.rmtree(CHROMA_DB_PATH)

        print("Old ChromaDB deleted.")

# ==========================================
# READ TXT FILE
# ==========================================

def read_txt_file(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()

# ==========================================
# READ PDF FILE
# ==========================================

def read_pdf_file(file_path):

    text = ""

    try:

        reader = PdfReader(file_path)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as error:

        print(f"Error reading PDF: {file_path}")
        print(error)

    return text

# ==========================================
# LOAD DOCUMENTS
# ==========================================

def load_documents(folder_path):

    documents = []

    for file_name in os.listdir(folder_path):

        file_path = os.path.join(
            folder_path,
            file_name
        )

        text = ""

        # TXT FILE
        if file_name.lower().endswith(".txt"):

            text = read_txt_file(file_path)

        # PDF FILE
        elif file_name.lower().endswith(".pdf"):

            text = read_pdf_file(file_path)

        else:
            continue

        # SKIP EMPTY FILES
        if not text.strip():

            print(
                f"Skipped empty file: {file_name}"
            )

            continue

        documents.append({
            "file_name": file_name,
            "text": text
        })

    return documents

# ==========================================
# CHUNK TEXT
# ==========================================

def chunk_text(
    text,
    chunk_size=800,
    overlap=150
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# ==========================================
# CREATE CHROMADB COLLECTION
# ==========================================

def create_chroma_collection():

    embedding_function = (
        embedding_functions
        .SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return collection

# ==========================================
# INGEST DOCUMENTS
# ==========================================

def ingest_documents():

    # RESET OLD DATABASE
    reset_chroma_db()

    print("\nLoading documents...\n")

    documents = load_documents(DOCS_FOLDER)

    if not documents:

        print("No valid documents found.")
        return

    collection = create_chroma_collection()

    ids = []
    texts = []
    metadatas = []

    total_chunks = 0

    for doc in documents:

        file_name = doc["file_name"]

        text = doc["text"]

        print(f"Ingesting: {file_name}")

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):

            ids.append(str(uuid.uuid4()))

            texts.append(chunk)

            metadatas.append({

                # SOURCE FILE
                "source": file_name,

                # CHUNK NUMBER
                "chunk_index": index,

                # FILE TYPE
                "document_type": (
                    file_name
                    .split(".")[-1]
                    .lower()
                ),

                # LOWERCASE FILE NAME
                "file_name": (
                    file_name.lower()
                ),

                # CHUNK SIZE
                "chunk_size": len(chunk)
            })

            total_chunks += 1

    # STORE IN CHROMADB
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )

    print("\n===================================")
    print("DOCUMENT INGESTION COMPLETED")
    print("===================================")

    print(
        f"Documents Processed : {len(documents)}"
    )

    print(
        f"Total Chunks Stored : {total_chunks}"
    )

# ==========================================
# MAIN ENTRY
# ==========================================

if __name__ == "__main__":
    ingest_documents()