# RAG-Integrated Agentic Chatbot 🚀 
A versatile chatbot that combines Retrieval-Augmented Generation (RAG) with agentic capabilities for intelligent query handling. Built for researchers, students, and professionals, it performs real-time searches, document Q&A, and summarization on diverse inputs like PDFs, YouTube videos, URLs, and academic papers.
This project was developed as part of the "Intro to Data Management" course at Khoury College of Computer Sciences.
Features ✨

## Multi-Mode Queries: General web searches, document Q&A, and summarization.
Real-Time Integrations: Supports SerpAPI (web), NewsAPI (news), PubMed, ArXiv, and Wikipedia via LangChain agents.
Document Processing: Handles PDFs (PyPDFLoader), YouTube videos (YoutubeLoader), web pages (WebBaseLoader), and ArXiv papers.
Efficient Retrieval: Uses FAISS vector database and HuggingFace embeddings for fast, context-aware responses.
Agentic Workflow: ZERO_SHOT_REACT_DESCRIPTION agent for zero-shot tool execution and reasoning.
User-Friendly UI: Interactive Streamlit interface for seamless uploads and interactions.
Performance: Average document retrieval in 0.5s; smart tool selection boosts relevance by ~30% (manual tests).

## Tech Stack 🛠️

Frontend: Streamlit
Backend/AI: LangChain, OpenAI/Groq LLM (configurable)
Embeddings & Vector DB: HuggingFace Embeddings, FAISS
Document Loaders: PyPDFLoader, YoutubeLoader, WebBaseLoader
Other: RecursiveCharacterTextSplitter for chunking, external APIs (SerpAPI, NewsAPI, etc.)

## Installation 📦

Clone the repo:
bashgit clone https://github.com/yourusername/rag-agentic-chatbot.git
cd rag-agentic-chatbot

Create a virtual environment (Python 3.10+ recommended):
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies:
bashpip install -r requirements.txt
Note: Create a requirements.txt file with: streamlit, langchain, faiss-cpu, huggingface-hub, pypdf, youtube-transcript-api, beautifulsoup4, etc.
Set up API keys: Create a .env file with your keys (e.g., SERPAPI_API_KEY=your_key, NEWSAPI_API_KEY=your_key).

## Usage ▶️

Run the app:
bashstreamlit run app.py  # Assuming your main file is app.py

Open in browser: http://localhost:8501
Interact:

Upload PDFs/URLs/YouTube links.
Select mode: General Search, Document Q&A, or Summarization.
Query away! E.g., "Summarize this PDF on AI ethics" or "Latest news on climate change".

## Time Complexity & Data Structures 📊

Document Processing: O(n * m) – n docs, m avg length.
Vector Store: O(n * d) – n chunks, d embedding dim.
Query: General: O(t * q), Doc Q&A: O(k * log(n)).
Structures: FAISS (vector search), Dicts/Lists (caching), Recursive splitter.

## Limitations & Future Work ⚠️

Limitations: API dependencies (rate limits/downtime), English-only, memory-intensive for large docs, requires internet.
Future Ideas: Multilingual support, memory optimization, more APIs (e.g., Google Scholar), ML-enhanced tool selection, offline mode.
