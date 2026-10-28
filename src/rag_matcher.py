"""Retrieval-based resume / JD matching.

Pipeline: chunk the resume -> embed -> index. Chunk the JD into requirements.
For each requirement, retrieve the closest resume chunks. A requirement counts
as covered when its best chunk scores at or above ``threshold``.

The threshold is NOT universal: score distributions differ between embedding
models. Tune it on a few resume/JD pairs you can judge by hand.
"""
try:
    from .chunker import chunk_text
    from .embeddings import get_embedder
    from .vector_index import InMemoryVectorIndex
except ImportError:
    from chunker import chunk_text
    from embeddings import get_embedder
    from vector_index import InMemoryVectorIndex

DEFAULT_THRESHOLD = 0.6


def build_resume_index(resume_text: str, embedder) -> InMemoryVectorIndex:
    chunks = chunk_text(resume_text)
    index = InMemoryVectorIndex(embedder)
    index.add([c.text for c in chunks], [{"section": c.section, "chunk": c.index} for c in chunks])
    return index


def semantic_match(resume_text: str, jd_text: str, embedder=None,
                   top_k: int = 2, threshold: float = DEFAULT_THRESHOLD) -> dict:
    embedder = embedder or get_embedder("fastembed")
    index = build_resume_index(resume_text, embedder)
    requirements = chunk_text(jd_text)

    matches = []
    for req in requirements:
        hits = index.search(req.text, k=top_k)
        best = hits[0].score if hits else 0.0
        matches.append({
            "requirement": req.text,
            "score": round(best, 4),
            "covered": best >= threshold,
            "evidence": [{"text": h.text, "section": h.metadata.get("section", "other"),
                          "score": round(h.score, 4)} for h in hits],
        })

    covered = sum(m["covered"] for m in matches)
    return {
        "embedder": getattr(embedder, "name", type(embedder).__name__),
        "threshold": threshold,
        "requirements_total": len(matches),
        "requirements_covered": covered,
        "coverage": round(100 * covered / len(matches), 2) if matches else 0.0,
        "mean_similarity": round(sum(m["score"] for m in matches) / len(matches), 4) if matches else 0.0,
        "matches": matches,
        "gaps": [m["requirement"] for m in matches if not m["covered"]],
    }
