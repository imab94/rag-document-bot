# 🚀 RAG Document Chatbot with Groq & Gemini

An advanced **Retrieval-Augmented Generation (RAG)** based AI chatbot that allows users to chat with their own documents using **Groq LLM**, **Google Gemini**, and **ChromaDB** vector storage.

This project demonstrates how modern GenAI systems combine:

* LLMs
* Embeddings
* Vector Databases
* Semantic Search
* Conversational Memory

to build intelligent document-aware AI assistants.

---

# ✨ Features

* 📄 Upload and ingest documents
* 🔍 Semantic search using embeddings
* 🤖 Chat with documents using AI
* ⚡ Ultra-fast inference with Groq
* 🧠 Gemini integration support
* 🗂️ ChromaDB vector database
* 💬 Context-aware responses
* 🐍 Pure Python backend implementation

---

# 🛠️ Tech Stack

| Technology    | Purpose                    |
| ------------- | -------------------------- |
| Python        | Backend Development        |
| Groq API      | High-speed LLM inference   |
| Google Gemini | Generative AI model        |
| ChromaDB      | Vector Database            |
| LangChain     | RAG pipeline orchestration |
| Embeddings    | Semantic similarity search |

---

# 📂 Project Structure

```bash
rag_document_bot/
│
├── docs/                 # Documents for ingestion
├── ingest.py             # Document ingestion pipeline
├── chat_groq.py          # Groq chatbot integration
├── chat_gemini.py        # Gemini chatbot integration
├── requirements.txt      # Dependencies
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/imab94/rag-document-bot.git

cd rag-document-bot
```

---

## 2️⃣ Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_gemini_api_key
```

---

# 📥 Ingest Documents

Place your documents inside the `docs/` folder.

Then run:

```bash
python ingest.py
```

This will:

* read documents
* split text into chunks
* create embeddings
* store vectors in ChromaDB

---

# 💬 Run Chatbot

## Using Groq

```bash
python chat_groq.py
```

## Using Gemini

```bash
python chat_gemini.py
```

---

# 🧠 How RAG Works

```text
User Query
   ↓
Embedding Generation
   ↓
ChromaDB Vector Search
   ↓
Relevant Context Retrieval
   ↓
LLM (Groq / Gemini)
   ↓
AI Generated Response
```

---

# 📸 Example Use Cases

* AI PDF chatbot
* Company knowledge assistant
* Resume analyzer
* Research document assistant
* Internal enterprise search
* Personal AI knowledge base

---

# 🚀 Future Improvements

* Streamlit UI
* Multi-PDF support
* Chat history memory
* Authentication system
* Docker deployment
* Agentic AI workflows
* MCP Server integration
* Advanced prompt engineering

---

# 🤝 Contributing

Contributions are welcome.

Feel free to fork this repository and submit pull requests.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Arun Bhardwaj**

Passionate about:

* Generative AI
* RAG Systems
* Python Backend Development
* Agentic AI
* LLM Engineering

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.
