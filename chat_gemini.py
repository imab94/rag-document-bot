import os
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from chromadb.utils import embedding_functions


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please add it in .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "my_documents"


def get_chroma_collection():
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return collection


def retrieve_relevant_context(question, top_k=4):
    collection = get_chroma_collection()

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for i, doc in enumerate(documents):
        source = metadatas[i].get("source", "unknown")
        chunk_index = metadatas[i].get("chunk_index", "unknown")

        context_parts.append(
            f"""
Source: {source}
Chunk: {chunk_index}
Content:
{doc}
"""
        )

    return "\n\n".join(context_parts)


def generate_answer(question, context):
    prompt = f"""
You are a strict document-based AI assistant.

Very important rules:
1. Answer ONLY using the provided document context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT make assumptions.
5. If the answer is not clearly available in the provided context, reply exactly:
   "I don't have this information in the provided documents."
6. If the user asks for a specific format like bullet points, headings, table, short answer, or step-by-step format, follow that format.
7. Even while following the requested format, use ONLY the provided document context.
8. If some part is available and some part is not available, answer only the available part and clearly mention:
   "Some requested information is not available in the provided documents."
9. Answer in the same language as the user's question.
10. Mention source document name if available.

Document Context:
{context}

User Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=700
        )
    )

    return response.text


def ask_question(question):
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context)
    return answer


def main():
    print("Gemini RAG Document Chatbot Started")
    print("Ask questions from your uploaded documents.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ")

        if question.lower() in ["exit", "quit", "stop"]:
            print("Bot stopped.")
            break

        answer = ask_question(question)

        print("\nBot:")
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()