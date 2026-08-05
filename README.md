# World Data AI (RAG Engine) 🌍

A high-performance **Retrieval-Augmented Generation (RAG)** application built to intelligently query and converse with massive external datasets (PDFs) without hallucinations. 

This project demonstrates a production-grade architecture combining state-of-the-art open-source Large Language Models with local embeddings and a Vector Database.

## 🧠 Architecture & Tech Stack
*   **Framework:** [LangChain](https://python.langchain.com/) (LCEL - LangChain Expression Language)
*   **LLM Inference:** LLaMA-3 (8B) deployed via Groq for ultra-low latency response times.
*   **Vector Database:** `DocArrayInMemorySearch` for rapid similarity search and document retrieval.
*   **Embeddings:** `GPT4AllEmbeddings` running entirely locally to reduce API costs and improve privacy.
*   **Frontend:** Streamlit for a clean, interactive chat UI.

## ⚙️ How it Works (The RAG Pipeline)
1.  **Ingestion:** `PyPDFLoader` ingests structured data (`world_data.pdf`).
2.  **Chunking:** `RecursiveCharacterTextSplitter` processes the data into semantic chunks (size: 1000, overlap: 200).
3.  **Embedding:** Text is embedded using GPT4All and stored in the Vector Database.
4.  **Retrieval:** When a user asks a question, the engine retrieves the most contextually relevant chunks.
5.  **Generation:** The context is passed into a custom prompt template and fed into LLaMA-3 to generate a precise, hallucination-free response.

## 🚀 Running Locally
1. Clone the repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Create a `.env` file and add your Groq API key: `GROQ_API_KEY=your_key_here`
4. Run the Streamlit server: `streamlit run app.py`

---
*Built by Rohit Singh — AI/ML Engineering Student & Team Lead at Code Titans.*
