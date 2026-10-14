# Architecture

## Components

### Parser (`parser.py`)
Extracts raw text from PDF (pdfplumber) and text files, and splits it into sections by heading.

### Analyzer (`analyzer.py`)
Rule-based skill extraction using regex over a 45-skill vocabulary. Computes matched skills,
missing skills and a score (matched JD skills / JD skills).

### Matcher (`matcher.py`)
Word-count cosine similarity and TF-IDF cosine similarity. `rank_resumes` fits TF-IDF on a set of
resumes plus the JD and ranks the resumes.

### CLI (`main.py`)
Reads files, calls the modules above and prints text or JSON.

## Data flow

Resume/JD files -> Parser -> Analyzer (skills, score) + Matcher (similarity) -> text / JSON output
