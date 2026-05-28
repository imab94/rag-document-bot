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
        You are an intelligent document-based AI assistant.

        Your task is to understand the provided document context and generate clear,
        natural, professional, and human-like answers.

        IMPORTANT RULES:

        1. Use ONLY the provided document context for answering.
        2. Do NOT use outside knowledge, internet knowledge, or assumptions.
        3. First understand the context, then explain the answer naturally in your own words.
        4. Do NOT blindly copy exact lines from the documents unless necessary.
        5. If multiple context chunks are provided, combine the information intelligently.
        6. Keep answers concise, relevant, clean, and well-structured.
        7. Follow the exact format requested by the user.

        Examples:
        - "5 things" → provide EXACTLY 5 numbered or bullet points
        - "3 reasons" → provide EXACTLY 3 numbered or bullet points
        - "top 10" → provide EXACTLY 10 numbered points
        - "list" → always use bullet points
        - "steps" → always use numbered steps

        Never convert requested points into large paragraphs.
        Keep each point separate and clearly formatted.

        8. If the answer is partially available, answer only from the available information.
        9. If the answer is not available in the documents, reply exactly:
        "I don't have enough information in the provided documents."

        10. If code examples are present in the context, explain them clearly in your own words.
        11. Never hallucinate, invent facts, or generate unsupported information.
        12. Answer in the SAME language as the user's question.
            - English question → English answer
            - Hindi question → Hindi answer
            - Hinglish question → Hinglish answer

        13. Do NOT repeat the user's question in the answer.
        14. Do NOT copy full paragraphs unless required.
        15. Generate conversational and professional responses.

        Your goal is to:
        - retrieve information
        - understand information
        - generate intelligent answers
        - sound natural and helpful
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
        temperature=0.2,
        max_tokens=700
    )

    return response.choices[0].message.content


def extract_unique_sources(context):

    sources = set()

    for line in context.splitlines():

        if line.strip().startswith("Source:"):

            source_name = line.replace(
                "Source:",
                ""
            ).strip()

            sources.add(source_name)

    return sorted(list(sources))


def ask_question(question):
    context = retrieve_relevant_context(question)
    answer = generate_answer(question, context)
    unique_sources = extract_unique_sources(context)

    if unique_sources:
        answer += "\n\nSources:\n"

        for source in unique_sources:
            answer += f"- {source}\n"
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