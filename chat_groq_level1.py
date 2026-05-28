import os
import chromadb

from dotenv import load_dotenv
from groq import Groq
from chromadb.utils import embedding_functions

from utils import (
    detect_metadata_filter,
    clean_response
)

# ==========================================
# LOAD ENV VARIABLES
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env file."
    )

# ==========================================
# GROQ CLIENT
# ==========================================

client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# CHROMADB SETTINGS
# ==========================================

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "my_documents"

# ==========================================
# CHAT MEMORY
# ==========================================

chat_history = []

# ==========================================
# CHROMADB COLLECTION
# ==========================================

def get_chroma_collection():

    embedding_function = (
        embedding_functions
        .SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return collection

# ==========================================
# QUERY REWRITING
# ==========================================

def rewrite_query(question):

    rewrite_prompt = f"""
Rewrite the following user query into a clean,
clear, detailed search query for document retrieval.

User Query:
{question}

Rewritten Query:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt
            }
        ],
        temperature=0
    )

    rewritten_query = (
        response.choices[0]
        .message.content
        .strip()
    )

    return rewritten_query

# ==========================================
# RETRIEVE CONTEXT
# ==========================================

def retrieve_relevant_context(question, top_k=4):

    collection = get_chroma_collection()

    rewritten_query = rewrite_query(question)

    metadata_filter = detect_metadata_filter(
        question
    )

    query_params = {
        "query_texts": [rewritten_query],
        "n_results": top_k
    }

    if metadata_filter:
        query_params["where"] = metadata_filter

    results = collection.query(**query_params)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for i, doc in enumerate(documents):

        source = metadatas[i].get(
            "source",
            "unknown"
        )

        chunk_index = metadatas[i].get(
            "chunk_index",
            "unknown"
        )

        chunk_preview = " ".join(
            doc[:120].split()
        )

        context_parts.append(
            f"""
==================================================
SOURCE DOCUMENT: {source}
CHUNK NUMBER: {chunk_index}

CHUNK PREVIEW:
{chunk_preview}...

CONTENT:
{doc}
==================================================
"""
        )

    context = "\n\n".join(context_parts)

    return context

# ==========================================
# EXTRACT SOURCES
# ==========================================

def extract_unique_sources(context):

    sources = set()

    for line in context.splitlines():

        if "SOURCE DOCUMENT:" in line:

            source_name = (
                line.replace(
                    "SOURCE DOCUMENT:",
                    ""
                )
                .strip()
            )

            sources.add(source_name)

    return sorted(list(sources))

# ==========================================
# GENERATE ANSWER
# ==========================================

def generate_answer(question, context):

    conversation_context = ""

    for item in chat_history[-5:]:

        conversation_context += (
            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}\n\n"
        )

    system_prompt = """
You are an intelligent document-based AI assistant.

IMPORTANT RULES:

    1. Use ONLY the provided document context.
    2. Never use outside knowledge.
    3. Never hallucinate or invent facts.
    4. Explain naturally in your own words.
    5. Keep answers concise and well structured.
    6. Follow the user's requested format exactly.

    Examples:
    - "5 things" → EXACTLY 5 bullet points
    - "3 reasons" → EXACTLY 3 numbered points
    - "steps" → numbered steps
    - "list" → bullet points

    7. Never convert requested points into large paragraphs.
    8. If answer is unavailable, reply:
    "I don't have enough information in the provided documents."
    9. Answer in the same language as the user.
    10. Do not repeat the question.
    11. Use plain text only.
"""

    user_prompt = f"""
Previous Conversation:
{conversation_context}

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
        temperature=0.3,
        max_tokens=700
    )

    answer = (
        response.choices[0]
        .message.content
    )

    return clean_response(answer)

# ==========================================
# ASK QUESTION
# ==========================================

def ask_question(question):

    context = retrieve_relevant_context(question)

    answer = generate_answer(
        question,
        context
    )

    unique_sources = extract_unique_sources(
        context
    )

    if unique_sources:

        answer += "\n\nSources:\n"

        for source in unique_sources:

            answer += f"- {source}\n"

    # SAVE CHAT MEMORY
    chat_history.append({
        "question": question,
        "answer": answer
    })

    return answer

# ==========================================
# MAIN CHAT LOOP
# ==========================================

def main():

    print("Advanced Groq RAG Chatbot Started")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() in [
            "exit",
            "quit",
            "stop"
        ]:
            print("Bot stopped.")
            break

        answer = ask_question(question)

        print("\nBot:")
        print(answer)

        print("-" * 60)

# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()