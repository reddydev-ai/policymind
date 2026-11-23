"""Streamlit UI for PolicyMind. Upload policy PDFs, then ask questions and see
cited answers with a confidence score. Run with:  streamlit run streamlit_app.py
"""
import os
import secrets
import tempfile

import streamlit as st

from app.config import get_settings
from app.ingestion import ingest_pdf
from app.rag import answer_question

st.set_page_config(page_title="PolicyMind", page_icon="📄", layout="centered")
# Local Streamlit calls the ingestion/RAG code directly; it does not use FastAPI.
# Fail closed unless an operator explicitly enables this local-only demo.
settings = get_settings()
if os.getenv("POLICYMIND_ENABLE_LOCAL_UI", "").lower() != "true":
    st.error("Standalone UI is disabled by default. Use the authenticated API for controlled deployments.")
    st.stop()
if not settings.api_key:
    st.error("Configure POLICYMIND_API_KEY before enabling the local demo UI.")
    st.stop()
local_access_key = st.text_input("Local demo access key", type="password")
if not local_access_key or not secrets.compare_digest(local_access_key, settings.api_key):
    st.warning("An access key is required to run this demo.")
    st.stop()
st.warning("LOCAL DEMO ONLY: shared-key gate without user identity, document-level permissions or upload isolation. Do not expose publicly.")
st.title("📄 PolicyMind — Policy RAG Assistant")
st.caption("Hybrid search + reranking + cited answers. Internal policy documents only — not legal advice.")

with st.sidebar:
    st.header("1. Upload policy PDFs")
    files = st.file_uploader("Drop one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest", disabled=not files):
        with st.spinner("Indexing documents..."):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    path = tmp.name
                n = ingest_pdf(path)
                os.unlink(path)
                st.success(f"{f.name}: {n} chunks indexed")

st.header("2. Ask a question")
q = st.text_input("e.g. How many vacation days do new employees get?")
if st.button("Ask", disabled=not q):
    with st.spinner("Retrieving and answering..."):
        result = answer_question(q)
    st.markdown(f"**Answer**\n\n{result.answer}")
    st.progress(min(result.confidence, 1.0), text=f"Confidence: {result.confidence:.2f}")
    if result.citations:
        st.markdown("**Sources**")
        for c in result.citations:
            st.markdown(f"- `{c.source}` p.{c.page}  (relevance {c.score:.2f})")
    if not result.grounded:
        st.warning("Answered with low confidence — treat as 'not found in documents'.")
