"""CLI entry point for the resume analyzer."""
import argparse
import json
import sys

try:
    from .analyzer import analyze
    from .matcher import match, rank_resumes
    from .parser import extract_text, extract_sections
except ImportError:
    from analyzer import analyze
    from matcher import match, rank_resumes
    from parser import extract_text, extract_sections


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a resume with a job description")
    parser.add_argument("resume", nargs="+",
                        help="Path to resume (.pdf, .txt, or .md). Give several to rank them against the JD")
    parser.add_argument("job_description", help="Path to job description (.pdf, .txt, or .md)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON")
    args = parser.parse_args()

    if len(args.resume) > 1:
        try:
            jd_text = extract_text(args.job_description)
            resumes = {path: extract_text(path) for path in args.resume}
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        ranking = rank_resumes(resumes, jd_text)
        if args.as_json:
            print(json.dumps([{"resume": n, "tfidf_similarity": sc} for n, sc in ranking], indent=2))
        else:
            for i, (name, score) in enumerate(ranking, 1):
                print(f"{i}. {name}: {score:.2f}%")
        return 0

    try:
        resume = extract_text(args.resume[0])
        jd = extract_text(args.job_description)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not resume or not jd:
        print("Error: resume and job description must contain text.", file=sys.stderr)
        return 1

    analysis = analyze(resume, jd)
    sims = match(resume, jd)
    similarity = sims["similarity"]
    result = {
        **analysis,
        "text_similarity": similarity,
        "tfidf_similarity": sims["tfidf_similarity"],
        "resume_sections": extract_sections(resume),
        "job_sections": extract_sections(jd),
    }

    if args.as_json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"Overall skill match: {analysis['score']:.2f}%")
    print(f"Text similarity:      {similarity:.2f}% (word-count cosine)")
    print(f"TF-IDF similarity:    {sims['tfidf_similarity']:.2f}%")
    print("\nMatched skills:")
    print(", ".join(analysis["matched_skills"]) or "None")
    print("\nMissing skills:")
    print(", ".join(analysis["missing_skills"]) or "None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
