import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="eyeo Knowledge Base", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1rem; color: #666; margin-bottom: 1.5rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔍 eyeo Knowledge Base</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about Blockthrough billing, KPIs, data processes, and internal documentation.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📚 Documents Indexed")
    st.markdown("""
    - B2C Revenue & Subscriptions
    - Blockthrough Billing Docs
    - Business KPI Definitions
    - GAM Reports Process
    - Auction Flow & Latency
    - Google/Microsoft Billing
    - Programmatic Data Catalog
    - MAC to AAPV Relationships
    - Revenue Leakage Analysis
    - Monthly Billing Sheet Guide
    """)
    st.divider()
    num_sources = st.slider("Number of sources to retrieve", 2, 8, 4)
    st.divider()
    st.caption("Built by Akhil | PGP-AIML Project")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="./vectorstore", embedding_function=embeddings, collection_name="eyeo_docs")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if not GROQ_API_KEY:
    st.error("App not configured. Please contact Akhil.")
else:
    try:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": num_sources})
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.3)
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources"):
                        for s in msg["sources"]:
                            st.markdown(f"- {s}")
        
        if question := st.chat_input("Ask a question about eyeo documentation..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    docs = retriever.invoke(question)
                    context = format_docs(docs)
                    
                    template = """You are an expert assistant for eyeo/Blockthrough internal documentation.
Use the context below to answer the question. Be specific and reference actual table names, metrics, or processes when relevant.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
                    
                    prompt = PromptTemplate.from_template(template)
                    response = llm.invoke(prompt.format(context=context, question=question))
                    answer = response.content if hasattr(response, 'content') else str(response)
                    
                    st.markdown(answer)
                    
                    sources = []
                    seen = set()
                    for d in docs:
                        name = os.path.basename(d.metadata.get('source', 'Unknown'))
                        page = d.metadata.get('page', 0) + 1
                        src = f"{name} (page {page})"
                        if src not in seen:
                            seen.add(src)
                            sources.append(src)
                    
                    with st.expander("📚 Sources"):
                        for s in sources:
                            st.markdown(f"- {s}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

    except Exception as e:
        st.error(f"Error: {e}")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.session_state.messages:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
