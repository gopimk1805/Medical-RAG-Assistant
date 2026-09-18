# 🏥 Medical RAG Assistant

A medical question-answering system built using **Retrieval-Augmented Generation (RAG)**.

The application retrieves relevant information from a curated medical knowledge base and uses **Google Gemini** to generate clear, grounded responses.

---

## 🚀 Features

- 📄 Medical information extracted from PDF documents
- ✂️ Document chunking for efficient retrieval
- 🧠 Sentence Transformer embeddings
- 🔎 Semantic similarity search
- 🗄️ ChromaDB vector database
- 🤖 Google Gemini powered answer generation
- 📚 Source information displayed with answers
- 🛡️ Grounded responses using retrieved context
- 🚫 Handles questions outside the available knowledge base
- 🌐 Flask-based web interface
- 🔐 API key stored securely using `.env`

---

## 📚 Current Knowledge Base

The current knowledge base contains information about:

1. Influenza
2. Dengue
3. Malaria
4. Diabetes
5. Asthma
6. Hypertension

The information is stored as PDF documents and converted into searchable vector embeddings.

---

## 🏗️ System Architecture

```text
                 User Question
                       │
                       ▼
                Flask Web Interface
                       │
                       ▼
             Sentence Transformer
                Query Embedding
                       │
                       ▼
                  ChromaDB
             Semantic Retrieval
                       │
                       ▼
             Relevant Medical Chunks
                       │
                       ▼
              Grounded Prompt
                       │
                       ▼
               Google Gemini
                       │
                       ▼
              Generated Answer
                       │
                       ▼
          Answer + Sources + UI