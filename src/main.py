"""CLI entry point for the resume analyzer."""
import argparse
import json
import os
import sys

try:
    from .analyzer import analyze
    from .matcher import match, rank_resumes
    from .parser import extract_text, extract_sections
    from .rag_matcher import DEFAULT_THRESHOLD, semantic_match
    from .embeddings import get_embedder
    from .explainer import anthropic_llm, explain
except ImportError:
    from analyzer import analyze
    from matcher import match, rank_resumes
    from parser import extract_text, extract_sections
    from rag_matcher import DEFAULT_THRESHOLD, semantic_match
    from embeddings import get_embedder
    from explainer import anthropic_llm, explain


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a resume with a job description")
    parser.add_argument("resume", nargs="+",
                        help="Path to resume (.pdf, .txt, or .md). Give several to rank them against the JD")
    parser.add_argument("job_description", help="Path to job description (.pdf, .txt, or .md)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable JSON")
    parser.add_argument("--semantic", action="store_true",
                        help="Also run embedding-based retrieval matching (RAG pipeline)")
    parser.add_argument("--embedder", choices=["fastembed", "hashing"], default="fastembed",
                        help="Embedding backend for --semantic (hashing is offline and lexical)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Similarity needed to count a JD requirement as covered")
    parser.add_argument("--explain", action="store_true",
                        help="Add a written explanation (uses an LLM if ANTHROPIC_API_KEY is set, else a template). Implies --semantic")
    args = parser.parse_args()
    if args.explain:
        args.semantic = True

    if len(args.resume) > 1:
        try:
            jd_text = extract_text(args.job_description)
            resumes = {path: extract_text(path) for path in args.resume}
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.semantic:
            try:
                embedder = get_embedder(args.embedder)
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            sem = {n: semantic_match(t, jd_text, embedder, threshold=args.threshold) for n, t in resumes.items()}
            ordered = sorted(sem.items(), key=lambda kv: (kv[1]["coverage"], kv[1]["mean_similarity"]), reverse=True)
            if args.as_json:
                print(json.dumps([{"resume": n, "coverage": r["coverage"], "mean_similarity": r["mean_similarity"]}
                                  for n, r in ordered], indent=2))
            else:
                for i, (name, r) in enumerate(ordered, 1):
                    print(f"{i}. {name}: coverage {r['coverage']:.2f}% (mean similarity {r['mean_similarity']:.2f})")
            return 0
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

    if args.semantic:
        try:
            semantic = semantic_match(resume, jd, get_embedder(args.embedder), threshold=args.threshold)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        result["semantic"] = semantic
        if args.explain:
            llm = None
            if os.environ.get("ANTHROPIC_API_KEY"):
                try:
                    llm = anthropic_llm()
                except RuntimeError:
                    llm = None
            result["explanation"] = explain(semantic, llm)

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
    if args.semantic:
        sem = result["semantic"]
        print(f"\nSemantic coverage ({sem['embedder']}, threshold {sem['threshold']}): "
              f"{sem['requirements_covered']}/{sem['requirements_total']} requirements "
              f"({sem['coverage']:.2f}%), mean similarity {sem['mean_similarity']:.2f}")
        print("\nRequirements without resume evidence:")
        print("\n".join(f"- {g}" for g in sem["gaps"]) or "None")
    if "explanation" in result:
        print(f"\nExplanation [{result['explanation']['source']}]:")
        print(result["explanation"]["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
