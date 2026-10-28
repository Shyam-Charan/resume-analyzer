"""Resume/JD text extraction utilities."""
from pathlib import Path
import re


def extract_text(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    if file_path.suffix.lower() == ".pdf":
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    return file_path.read_text(encoding="utf-8", errors="ignore").strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


SECTION_HEADINGS = {
    "summary": r"summary|profile|objective",
    "skills": r"technical skills|skills|technologies",
    "experience": r"experience|work experience|internship",
    "education": r"education|academic background",
    "projects": r"projects|academic projects|personal projects",
    "achievements": r"achievements|awards|honors",
}


def detect_heading(line: str) -> str | None:
    """Return the section key if ``line`` is a section heading, else None."""
    compact = re.sub(r"[^a-z ]", "", line.lower()).strip()
    return next((key for key, pattern in SECTION_HEADINGS.items()
                 if re.fullmatch(pattern, compact)), None)


def extract_sections(text: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = {key: "" for key in SECTION_HEADINGS}
    current = None
    for line in lines:
        matched = detect_heading(line)
        if matched:
            current = matched
            continue
        if current:
            result[current] += (" " if result[current] else "") + line
    return result
