"""Skill extraction and resume/JD analysis."""
import re

DEFAULT_SKILLS = [
    "python", "java", "c++", "c", "javascript", "typescript", "go", "sql",
    "html", "css", "react", "node.js", "express", "flask", "fastapi",
    "docker", "kubernetes", "aws", "azure", "git", "linux", "rest api",
    "machine learning", "deep learning", "nlp", "pytorch", "tensorflow",
    "scikit-learn", "pandas", "numpy", "llm", "rag", "langchain", "langgraph",
    "vector database", "mongodb", "mysql", "postgresql", "redis", "spark",
    "data structures", "algorithms", "oop", "operating systems", "computer networks"
]


def extract_skills(text: str, skills=None) -> list[str]:
    skills = skills or DEFAULT_SKILLS
    lower = text.lower()
    found = []
    for skill in skills:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, lower):
            found.append(skill)
    return sorted(set(found), key=str.lower)


def analyze(resume_text: str, jd_text: str) -> dict:
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))
    matched = sorted(resume_skills & jd_skills, key=str.lower)
    missing = sorted(jd_skills - resume_skills, key=str.lower)
    score = round(100 * len(matched) / len(jd_skills), 2) if jd_skills else 0.0
    return {
        "score": score,
        "resume_skills": sorted(resume_skills, key=str.lower),
        "job_skills": sorted(jd_skills, key=str.lower),
        "matched_skills": matched,
        "missing_skills": missing,
    }
