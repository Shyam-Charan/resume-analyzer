import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzer import analyze, extract_skills
from matcher import cosine_similarity


def test_skill_extraction_is_case_insensitive():
    skills = extract_skills("Python, PYTHON, SQL and Machine Learning")
    assert "python" in skills
    assert "sql" in skills
    assert "machine learning" in skills
    assert skills.count("python") == 1


def test_analysis_reports_matches_and_missing():
    result = analyze("Python SQL Git", "Python SQL Docker")
    assert result["matched_skills"] == ["python", "sql"]
    assert result["missing_skills"] == ["docker"]
    assert result["score"] == 66.67


def test_cosine_similarity_bounds():
    assert cosine_similarity("python sql", "python sql") == pytest.approx(1.0)
    assert cosine_similarity("python", "java") == 0.0


from matcher import idf, tfidf_similarity, rank_resumes


def test_idf_downweights_common_words():
    weights = idf(["python sql", "python docker", "python spark"])
    assert weights["python"] < weights["sql"]


def test_tfidf_similarity_bounds():
    assert tfidf_similarity("python sql", "python sql") == pytest.approx(1.0)
    assert tfidf_similarity("python", "java") == 0.0


def test_rank_resumes_puts_best_match_first():
    resumes = {
        "a.txt": "python pandas machine learning projects",
        "b.txt": "java spring hibernate backend services",
    }
    ranking = rank_resumes(resumes, "looking for python and machine learning")
    assert ranking[0][0] == "a.txt"
    assert ranking[0][1] > ranking[1][1]
