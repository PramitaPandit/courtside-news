import os
import random
import streamlit as st
import requests
import base64
import time
from dotenv import load_dotenv
from services.news_ingest import fetch_latest_news

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from pathlib import Path

BASE_DIR = Path(__file__).parent

#set_bg(str(BASE_DIR / "assets" / "grass_blur.png"))
#load_css(str(BASE_DIR / "styles" / "theme.css"))


load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_db"

def set_bg(image_path: str):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
              linear-gradient(to bottom, rgba(0,0,0,0.45), rgba(0,0,0,0.60)),
              url("data:image/png;base64,{data}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- LLM ----------------
def groq_chat(api_key: str, messages: list, system_msg: str,
              model: str, temperature: float) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_msg}] + messages,
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ---------------- Vector DB ----------------
@st.cache_resource
def load_db():
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    # Tag each doc so we can filter by type later
    for doc in docs:
        src = doc.metadata.get("source", "")
        doc.metadata["type"] = "news" if "news" in src else "knowledge"

    # Better chunking — smaller chunks, more overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    # Always re-index — clears stale data, picks up new articles
    vectordb._collection.delete(where={"type": {"$in": ["news", "knowledge"]}})
    if chunks:
        vectordb.add_documents(chunks)

    return vectordb


def load_css(path: str):
    with open(path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def unique_sources(docs):
    srcs = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        if src not in srcs:
            srcs.append(src)
    return srcs


# ---------------- Animation Helpers ----------------
def render_thinking_animation():
    return """
    <div class="tennis-stage">
      <div class="tennis-row">
        <div class="tennis-ball b1"></div>
        <div class="tennis-ball b2"></div>
        <div class="tennis-ball b3"></div>
      </div>
      <div class="thinking-text">🎾 Rallying the sources…</div>
    </div>
    """

def render_settled_court():
    return """
    <div style="margin: 8px 0 10px 0; border-radius: 12px; overflow:hidden; border:1px solid rgba(255,255,255,0.10);">
      <div style="
        height: 18px;
        background: repeating-linear-gradient(
          90deg,
          rgba(30,132,73,1) 0px,
          rgba(30,132,73,1) 10px,
          rgba(22,110,61,1) 10px,
          rgba(22,110,61,1) 20px
        );
        position: relative;">
        <div style="position:absolute; top:7px; left:0; right:0; height:2px; background:rgba(255,255,255,0.75);"></div>
        <div style="position:absolute; top:1px; right:14px; display:flex; gap:6px;">
          <div class="tennis-ball" style="width:14px;height:14px;"></div>
          <div class="tennis-ball" style="width:14px;height:14px;"></div>
          <div class="tennis-ball" style="width:14px;height:14px;"></div>
        </div>
      </div>
    </div>
    """

def random_tennis_error():
    phrases = [
        "🎾 Oops! An unforced error!",
        "💥 Double fault from the server!",
        "🏃‍♂️ That one sailed long!",
        "🎯 Just missed the baseline!",
        "🌪 A wild cross-court mishit!",
        "😬 Net cord… and it didn’t roll over!",
        "🔥 That rally broke down mid-point!",
        "👀 Hawkeye says… out!"
    ]
    return random.choice(phrases)


INTRO_LINES = [
    "Welcome to the World of Tennis.",
    "Where every question is a serve, every answer is a rally, and every conversation is match point.",
    "",
    "Built to serve and volley across the sport - ATP. WTA. Grand Slams. Rivalries. Rankings. Iconic moments.",
    "",
    "Step onto the court. Start a rally, and let’s play."
]

def render_intro_once():
    # Don’t replay on every rerun
    if st.session_state.get("intro_done", False):
        # Static version after it’s done
        st.markdown(
            "<div class='intro-wrap intro-card'>" +
            "".join([f"<div class='intro-line'>{line}</div>" if line else "<div style='height:8px'></div>"
                     for line in INTRO_LINES]) +
            "</div>",
            unsafe_allow_html=True
        )
        return

    holder = st.empty()

    for i, line in enumerate(INTRO_LINES):
        # Alternate left/right for “court entrance”
        direction_class = "slide-left" if i % 2 == 0 else "slide-right"

        # Build what’s been revealed so far
        revealed = []
        for j in range(i + 1):
            if INTRO_LINES[j] == "":
                revealed.append("<div style='height:8px'></div>")
            else:
                anim = ("slide-left" if j % 2 == 0 else "slide-right")
                revealed.append(f"<div class='intro-line {anim}'>{INTRO_LINES[j]}</div>")

        holder.markdown(
            "<div class='intro-wrap intro-card'>" + "".join(revealed) + "</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.22)  # speed of reveal

    st.session_state["intro_done"] = True

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(page_title="Tennis News Chatbot", page_icon="🎾", layout="wide")

    # CSS Styling
    #load_css("styles/theme.css")   # ✅ since it's inside /styles
    #st.write("BG exists:", os.path.exists("assets/grass_blur.png"))

    #set_bg("assets/grass_blur.png")

    load_css(str(BASE_DIR / "styles" / "theme.css"))

    bg_path = BASE_DIR / "assets" / "grass_blur.png"
    st.write("BG exists:", bg_path.exists())

    set_bg(str(bg_path))

    #st.title("🎾 Tennis News Chatbot")

    st.markdown("""
    <div class="title-wrapper">
        <div class="title-row">
            <div class="title-ball"></div>
            <h1 class="grand-title">CourtSide News</h1>
        </div>
        <div class="sub-title">AI-powered Tennis intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    

    render_intro_once()

    st.markdown(
        "<div class='howto-line'>Ask about players, rankings, matches, or recent headlines.</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🔥 Latest tennis news"):
        st.session_state.messages.append({"role": "user", "content": "What are the latest tennis headlines today?"})
        st.rerun()
    if col2.button("🏆 Rankings"):
        st.session_state.messages.append({"role": "user", "content": "Show me the current ATP and WTA top 10 rankings."})
        st.rerun()
    if col3.button("🎾 Rivalries"):
        st.session_state.messages.append({"role": "user", "content": "Give me the biggest current rivalries on ATP and WTA tours."})
        st.rerun()
    if col4.button("📅 Tournaments"):
        st.session_state.messages.append({"role": "user", "content": "What are the next big tournaments coming up and who are the favorites?"})
        st.rerun()


    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found.")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Settings")

        # --- News refresh ---
        st.subheader("📰 News")
        if st.button("🔄 Refresh news"):
            with st.spinner("Fetching latest articles..."):
                count = fetch_latest_news(max_items_per_feed=10, hours=48)
                st.cache_resource.clear()   # force reload of vector DB
                st.success(f"✅ {count} new articles fetched!")
                st.rerun()

        show_advanced = st.toggle("Show advanced", value=False)

        if show_advanced:
            st.subheader("Advanced Settings")
            model = st.selectbox(
                "Groq model",
                ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
                index=0
            )
            top_k = st.slider("Retrieve chunks (k)", 2, 8, 4)
            temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1)
        else:
            # defaults when hidden
            model = "llama-3.1-8b-instant"
            top_k = 4
            temperature = 0.2

        if st.button("🧹 Clear chat"):
            st.session_state.messages = []
            st.rerun()

    vectordb = load_db()
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey! Ask me anything about Tennis 🎾"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a tennis question…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()


    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        question = st.session_state.messages[-1]["content"]

        with st.chat_message("assistant"):
            anim = st.empty()
            anim.markdown(render_thinking_animation(), unsafe_allow_html=True)

            docs = retriever.invoke(question)
            context = "\n\n".join([d.page_content for d in docs])

            #system = (   
            #    "You are a tennis assistant. Use context if available. "
            #    "If insufficient, use general knowledge. Be concise. "
            #    "If the user asks for latest news, only answer using fetched articles or say you need to fetch."
            #    "Cite sources when possible."
            #)

            system = """You are CourtSide, an expert tennis analyst and journalist.
            Rules:
            - For recent news questions: answer ONLY from the provided context. If no context matches, say: "I don't have a recent article on that — try clicking Refresh News in the sidebar."
            - For general tennis knowledge (rules, history, rankings structure): answer from your training knowledge.
            - Always mention player names, tournament names, and dates precisely.
            - Cite sources at the end as: Source: [filename]
            - Keep answers concise — under 200 words unless asked to elaborate.
            - If asked about rankings, note that your data may not be live.
            """

            #user = f"Question: {question}\n\nContext:\n{context}"

            # Add context to the last user message
            messages_with_context = st.session_state.messages[:-1] + [{
                "role": "user",
                "content": f"{question}\n\nRelevant context:\n{context}"
                }]

            #answer = groq_chat(api_key, user, system, model=model, temperature=temperature)
            answer = groq_chat(api_key, messages_with_context, system, model=model, temperature=temperature)

            try:
                #answer = groq_chat(api_key, user, system, model=model, temperature=temperature)
                # NEW — sends full conversation history 
                answer = groq_chat(
                    api_key,
                    st.session_state.messages,   # full history
                    system,
                    model=model,
                    temperature=temperature
                    )
                
            except Exception as e:
                anim.empty()
                st.markdown(f"### {random_tennis_error()}")
                st.info("The server dropped the rally. Try again in a moment.")
                st.stop()

            #anim.markdown(render_settled_court(), unsafe_allow_html=True)
            #st.markdown(answer)
            anim.empty()  # remove bouncing animation
            st.markdown(answer)
            st.markdown(render_settled_court(), unsafe_allow_html=True)

            with st.expander("📚 Sources / Retrieved context"):
                for s in unique_sources(docs):
                    st.markdown(f"- `{s}`")

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
