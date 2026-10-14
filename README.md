# resume-analyzer

Command-line tool that compares a resume with a job description (JD) and reports
which skills match, which are missing, and how similar the two texts are.

## How it works

| Module | Role |
|---|---|
| `src/parser.py` | Extracts text from PDF / TXT / MD and splits it into sections (skills, experience, education, projects, ...) |
| `src/analyzer.py` | Rule-based skill extraction: regex match against a vocabulary of 45 technical skills; score = matched JD skills / total JD skills |
| `src/matcher.py` | Text similarity: word-count cosine and TF-IDF cosine (implemented from scratch, no ML library) |
| `src/main.py` | CLI, human-readable and JSON output |

The skill extraction is keyword matching, not a trained model, so it will miss
synonyms ("ML" vs "machine learning") and skills outside the vocabulary.

### TF-IDF

`idf = ln((1 + n) / (1 + df)) + 1` (smoothed), weight = term count x idf, then cosine similarity.
IDF needs several documents to be informative. With one resume and one JD it
barely differs from word counts, so pass several resumes to rank them:

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

- Fixed skill vocabulary and exact-phrase matching; no synonym handling.
- Experience-years and education extraction are not implemented.
- Similarity is lexical, not semantic (no embeddings).
