# Multilingual Conversational AI Assistant for Rural Advisory (RAG)

**Student:** Mudit Kumar (Roll No: M24DE3051)  
**Guide:** Dr. Abhishek Sarkar  
**Institution:** IIT Jodhpur  
**Date:** November 2025  

## 🌾 Project Overview
This project addresses the "Cognitive Last Mile" problem in Indian agriculture. While internet access has improved, critical information (government schemes, crop advisory) remains locked in complex English PDFs. 

This system is a **Bilingual (Hindi/English) Retrieval-Augmented Generation (RAG)** assistant that:
1.  **Ingests** agricultural handbooks (ICAR, KVK) and scheme guidelines.
2.  **Embeds** text using multilingual models to understand Hindi queries against English documents.
3.  **Retrieves** trustworthy context to prevent hallucinations.
4.  **Generates** answers using **Google Gemini Pro**, citing specific pages/sources.

## 🛠️ Technology Stack
* **Orchestration:** LangChain
* **LLM:** Google Gemini Pro (via API)
* **Embeddings:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
* **Vector Store:** FAISS (CPU Optimized)
* **Interface:** Streamlit

## 🚀 Setup Instructions

### 1. Prerequisites
* Python 3.10+
* Google AI Studio API Key (for Gemini)

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd kisan-sahayak-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
