# 📧 GenAI Cold Email Generator

A Generative AI tool built with **Llama 3.1**, **LangChain**, **ChromaDB**, and **Streamlit** to generate personalized cold emails for B2B client acquisition or job outreach.

---

## 🏗️ Project Architecture

1. **Scraping / Input**: Extract job details from a URL using `WebBaseLoader` or direct text input.
2. **LLM Extraction**: `ChatGroq` (Llama 3.1) extracts structured JSON containing `role`, `experience`, `skills`, and `description`.
3. **Vector Database**: `ChromaDB` performs semantic search on `my_portfolio.csv` to match required skills to portfolio case study links.
4. **Email Generation**: `ChatGroq` formats a tailored cold email incorporating matched portfolio links.
5. **Streamlit UI**: Clean web UI for easy interaction.

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Groq API Key
Ensure your `.env` file in the root folder contains your Groq API key:
```env
GROQ_API_KEY=gsk_...
```

### 3. Run Streamlit App
```bash
streamlit run app/main.py
```
---
