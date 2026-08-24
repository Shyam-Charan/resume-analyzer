"""CLI entry point for the resume analyzer."""
import argparse
from pathlib import Path
import sys

from analyzer import analyze
from matcher import match
from parser import extract_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a resume with a job description")
    parser.add_argument("resume", help="Path to resume (.pdf, .txt, or .md)")
    parser.add_argument("job_description", help="Path to job description (.pdf, .txt, or .md)")
    args = parser.parse_args()

    try:
        resume = extract_text(args.resume)
        jd = extract_text(args.job_description)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    analysis = analyze(resume, jd)
    similarity = match(resume, jd)["similarity"]
    print(f"Overall skill match: {analysis['score']:.2f}%")
    print(f"Text similarity:      {similarity:.2f}%")
    print("\nMatched skills:")
    print(", ".join(analysis["matched_skills"]) or "None")
    print("\nMissing skills:")
    print(", ".join(analysis["missing_skills"]) or "None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
