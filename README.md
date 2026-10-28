# resume-analyzer

Command-line tool that compares a resume with a job description (JD) and reports
which skills match, which are missing, and how similar the two texts are.

## How it works

| Module | Role |
|---|---|
| `src/parser.py` | Extracts text from PDF / TXT / MD and splits it into sections (skills, experience, education, projects, ...) |
| `src/analyzer.py` | Rule-based skill extraction: regex match against a vocabulary of 45 technical skills; score = matched JD skills / total JD skills |
| `src/matcher.py` | Text similarity: word-count cosine and TF-IDF cosine (implemented from scratch, no ML library) |
| `src/chunker.py` | Splits resumes and job descriptions into bullet-level, section-tagged chunks |
| `src/embeddings.py` | Embedding backends: FastEmbed (semantic) and a hashing embedder (offline, lexical) |
| `src/vector_index.py` | In-memory vector index with cosine-similarity top-k search (NumPy) |
| `src/rag_matcher.py` | Retrieval-based matching: finds resume evidence for each job requirement |
| `src/explainer.py` | Turns retrieval results into a written explanation, with an optional LLM |
| `src/main.py` | CLI, human-readable and JSON output |

The skill extraction is keyword matching, not a trained model, so it will miss
synonyms ("ML" vs "machine learning") and skills outside the vocabulary.

### TF-IDF

`idf = ln((1 + n) / (1 + df)) + 1` (smoothed), weight = term count x idf, then cosine similarity.
IDF needs several documents to be informative. With one resume and one JD it
barely differs from word counts, so pass several resumes to rank them:

## Semantic matching (RAG pipeline)

The keyword analysis above only sees exact skill names. The `--semantic` mode
adds a retrieval pipeline that works at the level of whole bullet points:

1. The resume is split into chunks (one per bullet or sentence, tagged with its section).
2. Each chunk is embedded and stored in an in-memory index.
3. Each job requirement is embedded and used as a query. The closest resume chunks are retrieved.
4. A requirement counts as covered when its best chunk scores at or above `--threshold`.
5. With `--explain`, the retrieved evidence is passed to an LLM that writes a short explanation.
   The prompt contains only the retrieved evidence and tells the model not to claim anything
   beyond it. Without an API key, a deterministic template explanation is used instead.

```bash
pip install -r requirements-rag.txt

# semantic coverage with real embeddings (downloads the model on first run)
python src/main.py data/sample_resume.txt data/sample_jd_ai.txt --semantic

# add a written explanation (set ANTHROPIC_API_KEY to use an LLM)
python src/main.py data/sample_resume.txt data/sample_jd_ai.txt --explain

# offline, no model download (lexical only, mainly for tests)
python src/main.py data/sample_resume.txt data/sample_jd_ai.txt --semantic --embedder hashing
```

Notes on the scores:

- The default model is `BAAI/bge-small-en-v1.5`. Cosine scores differ from model to model, so
  the default threshold of 0.6 is a starting point and should be tuned on resume and JD pairs
  you can judge by hand.
- The hashing embedder compares words, not meaning. It will not link "ML" with "machine learning".
- The LLM explanation is only as good as the retrieved evidence. The retrieval scores are always
  printed next to it so it can be checked.

## Usage

```bash
pip install -r requirements.txt

# one resume vs one JD
python src/main.py data/sample_resume.txt data/sample_jd.txt
python src/main.py data/sample_resume.txt data/sample_jd.txt --json

# rank several resumes against one JD (TF-IDF fitted on all of them)
python src/main.py resume_a.pdf resume_b.pdf resume_c.pdf job_description.pdf
```

## Tests

```bash
python -m pytest -q
```

## Limitations / next steps

- The keyword analysis uses a fixed skill vocabulary and exact-phrase matching, with no synonym handling.
- Experience-years and education extraction are not implemented.
- The semantic threshold is not calibrated against labelled data. There is no benchmark yet
  comparing keyword matching, TF-IDF and embeddings on resumes with known outcomes.
- Chunking assumes bullet-style resumes. Heavily formatted PDFs (columns, tables) may extract poorly.
