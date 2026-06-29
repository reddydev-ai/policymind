"""Streamlit UI for PolicyMind. Upload policy PDFs, then ask questions and see
cited answers with a confidence score. Run with:  streamlit run streamlit_app.py
"""
import os
import tempfile

import streamlit as st

from app.ingestion import ingest_pdf
from app.rag import answer_question

st.set_page_config(page_title="PolicyMind", page_icon="📄", layout="centered")
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
