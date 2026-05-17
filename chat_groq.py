import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from chromadb.utils import embedding_functions


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please add it in .env file.")

client = Groq(api_key=GROQ_API_KEY)

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
    system_prompt = """
You are a strict document-based AI assistant.

Very important rules:
1. Answer ONLY using the provided document context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT make assumptions.
5. If the answer is not clearly available in the provided context, reply exactly:
   "I don't have this information in the provided documents."
6. If the user asks for a specific format like bullet points, headings, table, short answer, or step-by-step format, follow that format.
7. Do not force labels like "Short Answer", "Detailed Answer", or "Summary" unless the user asks for that format.
8. Even while following the requested format, use ONLY the provided document context.
9. If some part is available and some part is not available, answer only the available part and clearly mention:
   "Some requested information is not available in the provided documents."
10. If the user asks for sample code, provide code only if code examples are present in the provided context.
11. If code examples are not present in the provided context, reply:
   "I don't have sample code in the provided documents."
12. Answer in the same language as the user's question.
13. Mention source document name if available.
"""

    user_prompt = f"""
Document Context:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        max_tokens=700
    )

    return response.choices[0].message.content


def ask_question(question):
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context)
    return answer


def main():
    print("Groq RAG Document Chatbot Started")
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