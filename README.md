# 🩺 Medical ChatBot

A **Retrieval-Augmented Generation (RAG)** based medical chatbot that answers questions using information retrieved from a curated medical knowledge base.

The application combines **Hugging Face embeddings**, **Pinecone vector search**, **Groq LLM inference**, **LangChain**, and **Flask** to provide concise, context-aware answers through a web interface.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 🚀 Features

* 📄 PDF-based medical knowledge ingestion
* 🧩 Document chunking and preprocessing
* 🔢 Hugging Face sentence embeddings
* 🗄️ Pinecone vector database
* 🔍 Semantic similarity search
* 🧠 Retrieval-Augmented Generation (RAG)
* ⚡ Fast inference using Groq
* 🤖 LLM-powered question answering
* 🌐 Flask web application
* 🔐 Environment-variable based API key management
* 🐍 Python 3.12 support
* 📦 Dependency management using `uv`

---

# 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │     Medical PDFs    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Document Loader   │
                    │      PyPDFLoader     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Splitter     │
                    │ RecursiveCharacter  │
                    │     TextSplitter    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Embeddings       │
                    │ Hugging Face Model  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Pinecone       │
                    │   Vector Database   │
                    └──────────┬──────────┘
                               │
                         User Question
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Retriever       │
                    │   Top-K Documents   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Prompt + Context │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Groq LLM       │
                    │    GPT-OSS 20B      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Response       │
                    └─────────────────────┘
```

---

# 🧠 How RAG Works

The chatbot uses **Retrieval-Augmented Generation** rather than asking the LLM to answer solely from its pretrained knowledge.

The pipeline is:

```text
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
Search Pinecone
      │
      ▼
Retrieve Relevant Medical Documents
      │
      ▼
Add Documents to Prompt
      │
      ▼
Send Context + Question to LLM
      │
      ▼
Generate Answer
```

This allows the application to ground responses in the medical documents stored in the vector database.

---

# 📁 Project Structure

```text
Medical-Chatbot/
│
├── app.py
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── helper.py
│   └── prompt.py
│
├── templates/
│   └── chat.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── research/
    └── *.ipynb
```

---

# ⚙️ Step 01 — Create Environment

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environment and dependency management.

Initialize the project:

```bash
uv init
```

Create a virtual environment:

```bash
uv venv .venv
```

Activate the environment:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Verify Python:

```bash
python --version
```

Recommended:

```text
Python 3.12.x
```

---

# 📦 Step 02 — Install Requirements

Install all dependencies:

```bash
uv pip install -r requirements.txt
```

Alternatively, if you're using `pyproject.toml`:

```bash
uv sync
```

---

# 🔑 Step 03 — Configure Environment Variables

Create a `.env` file in the root directory:

```bash
touch .env
```

Add your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

### ⚠️ Important

Never commit `.env` to GitHub.

Your `.gitignore` should contain:

```gitignore
.env
.env.*
!.env.example
.venv/
__pycache__/
.ipynb_checkpoints/
```

Create `.env.example` for other developers:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

---

# 🤗 Step 04 — Hugging Face Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

for generating vector embeddings.

The embedding model converts text into numerical vectors:

```text
Medical Text
     │
     ▼
Hugging Face Embedding Model
     │
     ▼
[0.12, -0.34, 0.87, ...]
```

The model generates **384-dimensional embeddings**.

Example:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

# 🗄️ Step 05 — Pinecone

Pinecone is used as the vector database.

Create a Pinecone index with the appropriate embedding dimension.

For:

```text
sentence-transformers/all-MiniLM-L6-v2
```

use:

```text
Dimension: 384
Metric: cosine
```

The application loads the existing index:

```python
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medicalbot",
    embedding=embeddings
)
```

---

# 🔍 Step 06 — Retriever

The Pinecone vector store is converted into a retriever:

```python
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

This retrieves the **top 3 most relevant documents** for each question.

For example:

```text
Question:
"What is Acromegaly and gigantism?"

             │
             ▼

       Pinecone Search

             │
       ┌─────┼─────┐
       ▼     ▼     ▼
     Doc 1 Doc 2 Doc 3
       │     │     │
       └─────┼─────┘
             ▼
          LLM Context
```

---

# ⚡ Step 07 — Groq

The chatbot uses Groq for fast LLM inference.

Example:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
```

`temperature=0` is used to make responses more deterministic.

---

# 📝 Step 08 — Prompt

The system prompt instructs the model to use retrieved context.

Example:

```python
system_prompt = """
You are an assistant for question-answering tasks.

Use the following pieces of retrieved context to answer the question.

If you don't know the answer, say that you don't know.

Use three sentences maximum and keep the answer concise.

Context:
{context}
"""
```

The question is then passed as:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
```

---

# 🔗 RAG Chain

The application connects the retriever, prompt, and LLM:

```text
                  User Question
                       │
                       ▼
                  Retriever
                       │
                       ▼
              Relevant Documents
                       │
                       ▼
                  Format Docs
                       │
                       ▼
              Prompt + Context
                       │
                       ▼
                   Groq LLM
                       │
                       ▼
                    Answer
```

A runnable-based implementation can be structured as:

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

The chain can then be invoked using:

```python
response = rag_chain.invoke(
    "What is Acromegaly and gigantism?"
)

print(response)
```

---

# 🌐 Flask Application

The web application is built using Flask.

Run:

```bash
python app.py
```

The application will start on:

```text
http://localhost:8080
```

You can also access it from another device on the same network using your machine's IP address:

```text
http://YOUR_IP_ADDRESS:8080
```

---

# 🔌 API Endpoint

The application provides a chat endpoint:

```text
POST /get
```

The question is sent using the `msg` form parameter.

Example:

```text
POST /get

msg=What is Acromegaly and gigantism?
```

The Flask application:

```text
Browser
   │
   ▼
POST /get
   │
   ▼
User Question
   │
   ▼
RAG Pipeline
   │
   ├── Pinecone Retrieval
   │
   ├── Context Construction
   │
   ├── Prompt
   │
   └── Groq LLM
   │
   ▼
Generated Answer
   │
   ▼
Browser
```

---

# 📚 Medical Knowledge Base

The chatbot is designed to answer questions from a medical knowledge base.

Typical workflow:

```text
Medical PDF
    │
    ▼
PDF Loader
    │
    ▼
Text Extraction
    │
    ▼
Text Chunking
    │
    ▼
Embedding Generation
    │
    ▼
Pinecone
```

For a new document collection, the documents need to be embedded and uploaded to the Pinecone index before the chatbot can retrieve them.

---

# 🛠️ Useful Commands

### Create environment

```bash
uv venv .venv
```

### Activate environment

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
uv pip install -r requirements.txt
```

### Check installed packages

```bash
uv pip list
```

### Check dependency conflicts

```bash
uv pip check
```

### Run Flask

```bash
python app.py
```

### Run with uv

```bash
uv run python app.py
```

---

# 🧪 Testing

Test the embedding model:

```python
from src.helper import download_hugging_face_embeddings

embeddings = download_hugging_face_embeddings()

print(
    embeddings.embed_query(
        "What is diabetes?"
    )
)
```

Test Pinecone retrieval:

```python
docs = retriever.invoke(
    "What is diabetes?"
)

for doc in docs:
    print(doc.page_content)
    print("-" * 80)
```

Test the complete RAG pipeline:

```python
response = rag_chain.invoke(
    "What is Acromegaly and gigantism?"
)

print(response)
```

---

# 🐛 Troubleshooting

## 1. `No module named 'langchain.chains'`

Newer LangChain versions have changed their package structure.

Use the runnable-based RAG approach described above instead of relying on:

```python
from langchain.chains import create_retrieval_chain
```

---

## 2. `No module named 'langchain.document_loaders'`

Use:

```python
from langchain_community.document_loaders import PyPDFLoader
```

instead of:

```python
from langchain.document_loaders import PyPDFLoader
```

---

## 3. `HuggingFaceEmbeddings` import error

Use:

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

Install:

```bash
uv pip install langchain-huggingface sentence-transformers
```

---

## 4. Pinecone import errors

Use the standard Pinecone client:

```python
from pinecone import Pinecone, ServerlessSpec
```

For normal RAG applications, there is generally no need to use:

```python
from pinecone.grpc import PineconeGRPC
```

---

## 5. Check dependency conflicts

Run:

```bash
uv pip check
```

If a package conflict appears, resolve the reported version constraint rather than randomly upgrading or downgrading packages.

---

# 🔐 Security

Never expose API keys in source code.

### ❌ Don't do this

```python
GROQ_API_KEY = "gsk_xxxxxxxxxxxxx"
```

### ✅ Use `.env`

```env
GROQ_API_KEY=your_key
```

Then:

```python
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

---

# 📄 `.gitignore`

A basic `.gitignore` should include:

```gitignore
.venv/
.env
.env.*
!.env.example

__pycache__/
*.py[cod]

.ipynb_checkpoints/

.vscode/
.idea/

*.log

data/
uploads/
models/
```

---

# 📦 Main Technologies

| Technology            | Purpose                                  |
| --------------------- | ---------------------------------------- |
| Python                | Core programming language                |
| Flask                 | Web application                          |
| LangChain             | LLM/RAG orchestration                    |
| LangChain Community   | Document loaders and integrations        |
| LangChain HuggingFace | Embedding integration                    |
| Hugging Face          | Sentence embeddings                      |
| Sentence Transformers | Embedding model                          |
| Pinecone              | Vector database                          |
| Groq                  | LLM inference                            |
| PyPDF                 | PDF processing                           |
| python-dotenv         | Environment variables                    |
| uv                    | Python environment/dependency management |

---

# 📈 Future Improvements

Possible extensions include:

* 🔐 User authentication
* 💬 Conversation history
* 📚 Multiple medical knowledge bases
* 📑 Source citations
* 🔎 Improved hybrid search
* 🧠 Reranking
* 🩺 Medical entity extraction
* 🗣️ Voice-based interaction
* 🌍 Multilingual support
* 📊 RAG evaluation
* 🛡️ Guardrails and safety filters
* 📈 LangSmith observability
* 🚀 Docker deployment
* ☁️ Cloud deployment
* ⚡ Streaming responses
* 🧪 Automated evaluation datasets

---

# ⚠️ Medical Disclaimer

This chatbot is an **educational/research project**.

It should not be used as a replacement for:

* Doctors
* Qualified healthcare professionals
* Emergency medical services
* Clinical diagnosis
* Prescribed treatment
* Professional medical advice

Always consult a qualified healthcare professional for medical concerns.

---

# 👨‍💻 Development

Clone the repository:

```bash
git clone https://github.com/DeepTensor-3070/Medical-Chatbot.git
```

Enter the project:

```bash
cd Medical-Chatbot
```

Create the environment:

```bash
uv venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

Create `.env`:

```bash
touch .env
```

Add your credentials:

```env
PINECONE_API_KEY=your_key
GROQ_API_KEY=your_key
```

Run:

```bash
python app.py
```

Open:

```text
http://localhost:8080
```

---

# ⭐ Project Goal

The goal of this project is to demonstrate how **Generative AI + Retrieval-Augmented Generation + Vector Databases** can be combined to build a domain-specific question-answering system.

```text
                    MEDICAL CHATBOT

                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Retrieval        Knowledge       Generation
          │              │              │
      Pinecone        Medical PDFs     Groq LLM
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                       RAG
                         │
                         ▼
                Medical Assistant
```

---

## 📜 License

Add your preferred license here, for example:

```text
MIT License
```

If this project uses medical documents from third parties, make sure their licenses and redistribution terms permit inclusion in the project.