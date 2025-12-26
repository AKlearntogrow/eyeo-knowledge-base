"""
eyeo Knowledge Base - Production Streamlit App v2.0
"""

import streamlit as st
import os
import time
from datetime import datetime
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

APP_CONFIG = {
    "title": "eyeo Knowledge Base",
    "icon": "🔍",
    "description": "AI-powered search for internal documentation",
    "version": "2.0",
    "author": "Akhil",
    "default_sources": 4,
    "max_sources": 8,
    "min_sources": 2,
}

st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["icon"],
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.1rem; color: #666; margin-bottom: 1.5rem;}
    .source-card {background-color: #f8f9fa; border-left: 4px solid #1E88E5; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;}
    .footer {text-align: center; color: #888; font-size: 0.8rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_vectorstore(_embeddings):
    return Chroma(persist_directory="./vectorstore", embedding_function=_embeddings, collection_name="eyeo_docs")

@st.cache_resource(show_spinner=False)
def load_llm(api_key: str):
    return ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0.3)

def format_docs(docs) -> str:
    return "\n\n---\n\n".join([f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}\nPage: {doc.metadata.get('page', 0) + 1}\nContent: {doc.page_content}" for doc in docs])

def get_sources(docs) -> List[Dict]:
    sources, seen = [], set()
    for doc in docs:
        name = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page = doc.metadata.get('page', 0) + 1
        key = f"{name}:{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"name": name.replace('.pdf', '').replace('-', ' ').replace('_', ' '), "page": page})
    return sources

def generate_answer(question: str, retriever, llm) -> tuple:
    docs = retriever.invoke(question)
    context = format_docs(docs)
    template = """You are an expert assistant for eyeo/Blockthrough internal documentation.
Be specific and reference actual table names, metrics, or processes when relevant.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    prompt = PromptTemplate.from_template(template)
    response = llm.invoke(prompt.format(context=context, question=question))
    answer = response.content if hasattr(response, 'content') else str(response)
    return answer, get_sources(docs)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    num_sources = st.slider("Sources to retrieve", APP_CONFIG["min_sources"], APP_CONFIG["max_sources"], APP_CONFIG["default_sources"])
    st.divider()
    st.markdown("## 📚 Knowledge Base")
    with st.expander("Indexed Documents", expanded=False):
        st.markdown("- B2C Revenue & Subscriptions\n- Blockthrough Billing Docs\n- Business KPI Definitions\n- GAM Reports Process\n- Auction Flow & Latency\n- Google/Microsoft Billing\n- Programmatic Data Catalog\n- MAC to AAPV Relationships\n- Revenue Leakage Analysis\n- Monthly Billing Sheet Guide")
    st.divider()
    st.markdown("## 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        duration = datetime.now() - st.session_state.session_start
        st.metric("Duration", f"{duration.seconds // 60}m")
    st.divider()
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.rerun()
    st.markdown(f'<div class="footer"><strong>{APP_CONFIG["title"]}</strong> v{APP_CONFIG["version"]}<br>Built by {APP_CONFIG["author"]}<br>PGP-AIML | UT Austin</div>', unsafe_allow_html=True)

# Main content
st.markdown(f'<p class="main-header">{APP_CONFIG["icon"]} {APP_CONFIG["title"]}</p><p class="sub-header">{APP_CONFIG["description"]}</p>', unsafe_allow_html=True)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    st.error("⚠️ Application not configured. Please contact the administrator.")
    st.stop()

try:
    with st.spinner("Loading knowledge base..."):
        embeddings = load_embeddings()
        vectorstore = load_vectorstore(embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": num_sources})
        llm = load_llm(GROQ_API_KEY)
        doc_count = vectorstore._collection.count()
except Exception as e:
    st.error(f"Error loading knowledge base: {e}")
    st.stop()

# Stats cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div style="background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;"><div style="font-size: 1.8rem; font-weight: bold;">{doc_count}</div><div style="font-size: 0.85rem; opacity: 0.9;">Indexed Chunks</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;"><div style="font-size: 1.8rem; font-weight: bold;">16</div><div style="font-size: 0.85rem; opacity: 0.9;">Documents</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div style="background: linear-gradient(135deg, #FB8C00 0%, #EF6C00 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;"><div style="font-size: 1.8rem; font-weight: bold;">~1s</div><div style="font-size: 0.85rem; opacity: 0.9;">Avg Response</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Quick questions
if not st.session_state.messages:
    st.markdown("##### 💡 Try asking:")
    quick_questions = ["What is the difference between BT Demand and Pub Demand?", "How do we handle missing GAM reports?", "What is the definition of MACs?", "What tables contain B2C revenue data?", "What caused the AAPV decline?"]
    cols = st.columns(2)
    for i, q in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(f"📝 {q[:50]}...", key=f"quick_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Sources"):
                for src in msg["sources"]:
                    st.markdown(f"• **{src['name']}** — Page {src['page']}")

# Chat input
if question := st.chat_input("Ask a question about eyeo documentation..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.total_queries += 1
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            start_time = time.time()
            try:
                answer, sources = generate_answer(question, retriever, llm)
                elapsed = time.time() - start_time
                st.markdown(answer)
                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)} documents)"):
                        for src in sources:
                            st.markdown(f"• **{src['name']}** — Page {src['page']}")
                st.caption(f"⚡ Response generated in {elapsed:.1f}s")
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as e:
                st.error(f"Error generating response: {e}")
