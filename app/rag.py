"""RAG answer generation.

Takes reranked chunks, builds a grounded prompt, and asks GPT-4o to answer ONLY
from the provided context. If no OPENAI_API_KEY/client is available, it returns a
transparent extractive demo answer from the top retrieved context so local demos
can still validate the RAG flow without paid API calls.
"""
from __future__ import annotations
from functools import lru_cache

from pydantic import BaseModel

from .config import get_settings
from .retrieval import retrieve, RetrievedChunk

settings = get_settings()


@lru_cache
def _llm():
    if not settings.openai_api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        return ChatOpenAI(
            model=settings.chat_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )
    except Exception:
        return None


class Citation(BaseModel):
    source: str
    page: int
    score: float


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    grounded: bool


SYSTEM_PROMPT = (
    "You are PolicyMind, an assistant that answers questions about internal "
    "company policy documents. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say you do not have enough "
    "information in the provided documents. Never invent policy. Be concise and "
    "cite the source filename and page you used inline like [source p.X]."
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[{c.source} p.{c.page}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def _confidence_from_score(score: float) -> float:
    return 1 / (1 + pow(2.718281828, -score))


def answer_question_with_evidence(question: str) -> tuple[Answer, list[RetrievedChunk], str]:
    chunks = retrieve(question)
    top_score = chunks[0].score if chunks else -10.0
    confidence = _confidence_from_score(top_score)

    if not chunks or top_score < settings.min_context_score:
        return Answer(
            answer=(
                "I don't have enough information in the provided documents to "
                "answer that confidently. Please rephrase or upload the relevant policy."
            ),
            citations=[],
            confidence=round(confidence, 3),
            grounded=False,
        ), chunks, "abstained"

    citations = [Citation(source=c.source, page=c.page, score=round(c.score, 3)) for c in chunks]
    llm = _llm()
    if llm is None:
        top = chunks[0]
        return Answer(
            answer=(
                f"Based on the retrieved policy context, the relevant passage says: "
                f"{top.text[:700]} [{top.source} p.{top.page}]"
            ),
            citations=citations,
            confidence=round(confidence, 3),
            grounded=True,
        ), chunks, "extractive-demo"

    from langchain_core.prompts import ChatPromptTemplate  # lazy import
    context = _build_context(chunks)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    chain = prompt | llm
    resp = chain.invoke({"context": context, "question": question})
    text = str(resp.content).strip()

    return Answer(
        answer=text,
        citations=citations,
        confidence=round(confidence, 3),
        grounded=True,
    ), chunks, "llm"


def answer_question(question: str) -> Answer:
    """Public answer API preserves the original response shape."""
    result, _evidence, _mode = answer_question_with_evidence(question)
    return result
