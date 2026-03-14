# CourtSide News 🎾
> AI-powered tennis chatbot — RAG + Groq LLaMA + Streamlit

Ask about players, rankings, match results, Grand Slams, and live news.

## Features
- Real-time news ingestion from BBC Sport & ESPN Tennis RSS feeds
- RAG pipeline with ChromaDB + sentence-transformers
- Multi-turn conversation memory
- Groq LLaMA 3.1 for fast inference

## Setup
1. Clone: `git clone https://github.com/PramitaPandit/courtside-news`
2. Install: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Groq API key
4. Run: `streamlit run app.py`

## Deploy on Streamlit Cloud
Push to GitHub, connect at share.streamlit.io,
and add GROQ_API_KEY under App Settings → Secrets.