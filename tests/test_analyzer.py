import sys
from pathlib import Path

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
    assert cosine_similarity("python sql", "python sql") == 1.0
    assert cosine_similarity("python", "java") == 0.0
