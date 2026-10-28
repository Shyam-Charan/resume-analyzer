"""Explanation step: turn retrieval results into a readable match explanation.

The LLM is injected as a plain callable ``prompt -> text`` so the module has no
hard dependency on any provider. ``anthropic_llm`` is one ready-made provider.
Without an LLM, ``template_explanation`` produces a deterministic summary.

Grounding: the prompt contains only retrieved evidence and tells the model not
to claim anything beyond it. This reduces, but does not eliminate, the chance
of a model overstating the match, so the retrieval scores are always shown next
to the explanation.
"""
import os
from typing import Callable

LLMFn = Callable[[str], str]
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_REQUIREMENTS_IN_PROMPT = 15


def build_prompt(result: dict) -> str:
    lines = [
        "You are reviewing how well a resume matches a job description.",
        "Use ONLY the evidence below. Do not invent skills, projects or experience.",
        "If a requirement has weak or no evidence, say so plainly.",
        "",
        f"Coverage: {result['requirements_covered']}/{result['requirements_total']} requirements "
        f"(similarity threshold {result['threshold']}).",
        "",
        "Requirements and best resume evidence:",
    ]
    for m in result["matches"][:MAX_REQUIREMENTS_IN_PROMPT]:
        status = "COVERED" if m["covered"] else "GAP"
        lines.append(f"- [{status}, score {m['score']:.2f}] {m['requirement']}")
        for ev in m["evidence"][:1]:
            lines.append(f"    evidence ({ev['section']}): {ev['text']}")
    lines += [
        "",
        "Write a short explanation (under 150 words): strongest matches, biggest gaps, "
        "and the two most valuable changes to the resume.",
    ]
    return "\n".join(lines)


def template_explanation(result: dict) -> str:
    covered = [m for m in result["matches"] if m["covered"]]
    gaps = [m for m in result["matches"] if not m["covered"]]
    out = [f"{result['requirements_covered']} of {result['requirements_total']} job requirements "
           f"have resume evidence above the {result['threshold']} similarity threshold."]
    if covered:
        top = sorted(covered, key=lambda m: m["score"], reverse=True)[:3]
        out.append("Strongest matches:")
        out += [f"  - {m['requirement']} (score {m['score']:.2f})" for m in top]
    if gaps:
        weakest = sorted(gaps, key=lambda m: m["score"])[:3]
        out.append("Weakest areas:")
        out += [f"  - {m['requirement']} (score {m['score']:.2f})" for m in weakest]
    return "\n".join(out)


def explain(result: dict, llm: LLMFn | None = None) -> dict:
    """Return {"text": ..., "source": "llm" | "template"}. Falls back to the template on LLM errors."""
    if llm is not None:
        try:
            text = llm(build_prompt(result)).strip()
            if text:
                return {"text": text, "source": "llm"}
        except Exception as exc:  # network, auth, quota: never break the CLI over the explanation
            return {"text": template_explanation(result), "source": f"template (LLM failed: {type(exc).__name__})"}
    return {"text": template_explanation(result), "source": "template"}


def anthropic_llm(model: str | None = None, max_tokens: int = 400) -> LLMFn:
    """Build an LLM callable backed by the Anthropic API (needs ANTHROPIC_API_KEY)."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic is not installed. Run: pip install -r requirements-rag.txt") from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic()
    model = model or os.environ.get("RESUME_ANALYZER_LLM_MODEL", DEFAULT_MODEL)

    def call(prompt: str) -> str:
        reply = client.messages.create(model=model, max_tokens=max_tokens,
                                       messages=[{"role": "user", "content": prompt}])
        return "".join(block.text for block in reply.content if block.type == "text")

    return call
