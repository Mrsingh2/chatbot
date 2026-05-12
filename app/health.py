"""Startup sanity checks. Print WARNING lines for likely-stale state; never fatal."""
from app.config import SCOPE_TOPICS, load_predefined_qa

ALIGNMENT_THRESHOLD = 0.30


def _avg_qa_scope_alignment(pairs: list[dict]) -> float:
    if not pairs or not SCOPE_TOPICS:
        return 1.0
    from app.embedder import embed, cosine_similarity
    qa_embeds = embed([p["q"] for p in pairs])
    scope_embeds = embed(SCOPE_TOPICS)
    total = 0.0
    for qa_emb in qa_embeds:
        total += max(cosine_similarity(qa_emb, s) for s in scope_embeds)
    return total / len(qa_embeds)


def run_startup_checks() -> None:
    pairs = load_predefined_qa()
    if not pairs:
        print(
            "WARNING: No predefined Q&A pairs loaded from .env (PREDEFINED_QA_1, ...). "
            "Check that .env is in sync with .env.example."
        )
    else:
        alignment = _avg_qa_scope_alignment(pairs)
        if alignment < ALIGNMENT_THRESHOLD:
            print(
                f"WARNING: Predefined Q&A look misaligned with SCOPE_TOPICS "
                f"(avg max cosine = {alignment:.2f}, expected >= {ALIGNMENT_THRESHOLD:.2f}). "
                f"Your .env may be left over from a different corpus / domain."
            )

    from app.vector_store import collection_count
    try:
        chunks = collection_count()
        if chunks == 0:
            print(
                "WARNING: Vector store is empty (0 chunks). "
                "Drop PDFs into pdfs/ and run `python ingest.py --force`."
            )
    except Exception as e:
        print(f"WARNING: Could not check vector store ({e}).")
