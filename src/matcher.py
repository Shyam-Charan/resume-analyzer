"""Resume/job-description matching.

Two similarity measures are provided:

* ``cosine_similarity`` - cosine over raw word counts (bag of words).
* ``tfidf_similarity`` / ``rank_resumes`` - cosine over TF-IDF vectors, which
  down-weights words that appear in every document (e.g. "experience") and
  up-weights words that are distinctive.

TF-IDF is only meaningful when the corpus has several documents. With one
resume and one JD every shared word has the same document frequency, so
``rank_resumes`` (many resumes against one JD) is where it really helps.
"""
from collections import Counter
import math
import re


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text.lower())


def _cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    numerator = sum(a[t] * b[t] for t in set(a) & set(b))
    denominator = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return numerator / denominator if denominator else 0.0


def cosine_similarity(left: str, right: str) -> float:
    """Cosine similarity over raw word counts."""
    return _cosine(Counter(_tokens(left)), Counter(_tokens(right)))


def idf(documents: list[str]) -> dict[str, float]:
    """Smoothed inverse document frequency: ln((1 + n) / (1 + df)) + 1."""
    n = len(documents)
    df = Counter()
    for doc in documents:
        df.update(set(_tokens(doc)))
    return {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}


def tfidf_vector(text: str, weights: dict[str, float]) -> dict[str, float]:
    counts = Counter(_tokens(text))
    return {t: c * weights[t] for t, c in counts.items() if t in weights}


def tfidf_similarity(left: str, right: str, corpus: list[str] | None = None) -> float:
    """TF-IDF cosine similarity. IDF is learned from ``corpus`` (default: the two texts)."""
    weights = idf(corpus if corpus else [left, right])
    return _cosine(tfidf_vector(left, weights), tfidf_vector(right, weights))


def rank_resumes(resumes: dict[str, str], jd_text: str) -> list[tuple[str, float]]:
    """Rank several resumes against one JD using TF-IDF fitted on all of them."""
    weights = idf(list(resumes.values()) + [jd_text])
    jd_vec = tfidf_vector(jd_text, weights)
    scored = [(name, round(_cosine(tfidf_vector(text, weights), jd_vec) * 100, 2))
              for name, text in resumes.items()]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def match(resume_text: str, jd_text: str) -> dict:
    return {
        "similarity": round(cosine_similarity(resume_text, jd_text) * 100, 2),
        "tfidf_similarity": round(tfidf_similarity(resume_text, jd_text) * 100, 2),
    }
