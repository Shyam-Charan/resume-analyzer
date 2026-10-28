import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chunker import chunk_text, split_units
from embeddings import HashingEmbedder, get_embedder
from explainer import build_prompt, explain, template_explanation
from parser import detect_heading, extract_sections
from rag_matcher import semantic_match
from vector_index import InMemoryVectorIndex

RESUME = """Projects
- Built a recommendation engine in Python with cosine similarity and offline evaluation
- Developed an anomaly detection pipeline using scikit-learn and SQL
Experience
- Debugged STM32 firmware and verified fixes with fault-injection tests
"""

JD = """Responsibilities
- Build recommendation systems in Python with offline evaluation
- Deploy Kubernetes clusters for streaming workloads
"""


# ---- parser refactor keeps behaviour -------------------------------------
def test_detect_heading():
    assert detect_heading("Technical Skills") == "skills"
    assert detect_heading("PROJECTS:") == "projects"
    assert detect_heading("Built a Python project") is None


def test_extract_sections_unchanged():
    sections = extract_sections(RESUME)
    assert "recommendation engine" in sections["projects"]
    assert "STM32" in sections["experience"]


# ---- chunker --------------------------------------------------------------
def test_bullets_become_separate_chunks_with_sections():
    chunks = chunk_text(RESUME)
    assert [c.section for c in chunks] == ["projects", "projects", "experience"]
    assert chunks[0].text.startswith("Built a recommendation engine")
    assert [c.index for c in chunks] == [0, 1, 2]


def test_wrapped_pdf_lines_are_merged():
    units = split_units("Experience\n- Built a pipeline that ingests\nlarge documents and indexes them")
    assert units == [("experience", "Built a pipeline that ingests large documents and indexes them")]


def test_short_units_dropped_and_long_units_split():
    assert chunk_text("Skills\nPython\nSQL") == []
    long_unit = " ".join(["Sentence number %d has several words in it." % i for i in range(30)])
    chunks = chunk_text(long_unit, max_words=40)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 48 for c in chunks)


# ---- embeddings -----------------------------------------------------------
def test_hashing_embedder_is_deterministic_and_shaped():
    e = HashingEmbedder(dim=64)
    a, b = e.embed(["python sql"]), e.embed(["python sql"])
    assert a.shape == (1, 64)
    assert np.array_equal(a, b)


def test_get_embedder_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_embedder("nope")


# ---- vector index ---------------------------------------------------------
def test_index_returns_best_match_first_with_metadata():
    index = InMemoryVectorIndex(HashingEmbedder())
    index.add(["python machine learning model", "java spring backend service", "docker kubernetes deployment"],
              [{"id": 0}, {"id": 1}, {"id": 2}])
    hits = index.search("machine learning in python", k=2)
    assert hits[0].metadata["id"] == 0
    assert hits[0].score >= hits[1].score
    assert len(hits) == 2


def test_index_scores_identical_text_as_one_and_handles_empty():
    index = InMemoryVectorIndex(HashingEmbedder())
    assert index.search("anything") == []
    index.add(["python sql docker"])
    assert index.search("python sql docker")[0].score == pytest.approx(1.0, abs=1e-5)
    assert index.search("", k=1)[0].score == 0.0


def test_index_rejects_mismatched_metadata():
    with pytest.raises(ValueError):
        InMemoryVectorIndex(HashingEmbedder()).add(["a b c"], [{}, {}])


# ---- rag matcher ----------------------------------------------------------
def test_semantic_match_finds_covered_requirement_and_gap():
    result = semantic_match(RESUME, JD, HashingEmbedder(), threshold=0.4)
    by_req = {m["requirement"]: m for m in result["matches"]}
    covered = by_req["Build recommendation systems in Python with offline evaluation"]
    gap = by_req["Deploy Kubernetes clusters for streaming workloads"]
    assert covered["covered"] and "recommendation engine" in covered["evidence"][0]["text"]
    assert not gap["covered"]
    assert result["requirements_total"] == 2 and result["requirements_covered"] == 1
    assert result["coverage"] == 50.0
    assert result["gaps"] == ["Deploy Kubernetes clusters for streaming workloads"]


def test_semantic_match_handles_empty_inputs():
    result = semantic_match("", JD, HashingEmbedder())
    assert result["requirements_covered"] == 0
    empty_jd = semantic_match(RESUME, "", HashingEmbedder())
    assert empty_jd["coverage"] == 0.0 and empty_jd["requirements_total"] == 0


# ---- explainer ------------------------------------------------------------
def _result():
    return semantic_match(RESUME, JD, HashingEmbedder(), threshold=0.4)


def test_prompt_is_grounded_in_evidence():
    prompt = build_prompt(_result())
    assert "ONLY the evidence below" in prompt
    assert "recommendation engine" in prompt
    assert "GAP" in prompt and "COVERED" in prompt


def test_explain_uses_llm_when_given():
    out = explain(_result(), llm=lambda prompt: "  LLM says hello  ")
    assert out == {"text": "LLM says hello", "source": "llm"}


def test_explain_falls_back_when_llm_fails_or_is_missing():
    def broken(_):
        raise ConnectionError("offline")
    failed = explain(_result(), llm=broken)
    assert failed["source"].startswith("template (LLM failed: ConnectionError")
    assert "1 of 2 job requirements" in failed["text"]
    assert explain(_result())["source"] == "template"
    assert "Weakest areas" in template_explanation(_result())


# ---- CLI ------------------------------------------------------------------
def test_cli_semantic_json(tmp_path, monkeypatch, capsys):
    import main
    resume, jd = tmp_path / "r.txt", tmp_path / "j.txt"
    resume.write_text(RESUME)
    jd.write_text(JD)
    monkeypatch.setattr(sys, "argv", ["main", str(resume), str(jd), "--semantic", "--embedder", "hashing",
                                      "--threshold", "0.4", "--explain", "--json"])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main.main() == 0
    data = json.loads(capsys.readouterr().out)
    assert data["semantic"]["requirements_total"] == 2
    assert data["explanation"]["source"] == "template"
    assert "score" in data  # existing keyword analysis still present


def test_cli_without_semantic_flag_is_unchanged(tmp_path, monkeypatch, capsys):
    import main
    resume, jd = tmp_path / "r.txt", tmp_path / "j.txt"
    resume.write_text("Python SQL Git")
    jd.write_text("Python SQL Docker")
    monkeypatch.setattr(sys, "argv", ["main", str(resume), str(jd), "--json"])
    assert main.main() == 0
    data = json.loads(capsys.readouterr().out)
    assert "semantic" not in data and "explanation" not in data
    assert data["matched_skills"] == ["python", "sql"]


def test_cli_ranks_multiple_resumes_semantically(tmp_path, monkeypatch, capsys):
    import main
    a, b, jd = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "jd.txt"
    a.write_text(RESUME)
    b.write_text("Experience\n- Managed retail store inventory and staff schedules every week\n")
    jd.write_text(JD)
    monkeypatch.setattr(sys, "argv", ["main", str(a), str(b), str(jd), "--semantic", "--embedder", "hashing",
                                      "--threshold", "0.4", "--json"])
    assert main.main() == 0
    ranking = json.loads(capsys.readouterr().out)
    assert ranking[0]["resume"] == str(a)


# ---- FastEmbed wrapper (real model cannot be downloaded in CI/offline) -----
def test_fastembed_wrapper_uses_library_interface(monkeypatch):
    import types

    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            for t in texts:  # generator of numpy rows, like the real library
                yield np.array([len(t), 1.0, 0.0], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    from embeddings import FastEmbedEmbedder
    e = FastEmbedEmbedder("fake-model")
    out = e.embed(["ab", "abcd"])
    assert out.shape == (2, 3) and out.dtype == np.float32
    assert e.embed([]).shape[0] == 0


def test_fastembed_missing_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # makes `import fastembed` raise ImportError
    from embeddings import FastEmbedEmbedder
    with pytest.raises(RuntimeError, match="requirements-rag.txt"):
        FastEmbedEmbedder()
