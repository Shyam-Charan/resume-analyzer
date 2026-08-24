"""Resume/job-description matching."""
from collections import Counter
import math
import re


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text.lower())


def cosine_similarity(left: str, right: str) -> float:
    a, b = Counter(_tokens(left)), Counter(_tokens(right))
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    numerator = sum(a[t] * b[t] for t in common)
    denominator = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
    return numerator / denominator if denominator else 0.0


def match(resume_text: str, jd_text: str) -> dict:
    return {"similarity": round(cosine_similarity(resume_text, jd_text) * 100, 2)}
