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

### Chunker (`chunker.py`)
Splits text into bullet-level chunks, merges lines that wrapped in PDF extraction, and tags each
chunk with its section. Very short units are dropped and very long ones are split on sentences.

### Embeddings (`embeddings.py`)
`FastEmbedEmbedder` produces semantic vectors. `HashingEmbedder` produces deterministic hashed
bag-of-words vectors for offline use and tests. Both expose `embed(texts) -> ndarray`.

### Vector index (`vector_index.py`)
Stores L2-normalised vectors in a NumPy matrix. Search is one matrix-vector product followed by a
top-k sort, which is enough for the few dozen chunks in a resume.

### RAG matcher (`rag_matcher.py`)
Builds the resume index, queries it with each job requirement and reports coverage, the best
evidence per requirement and the requirements with no evidence.

### Explainer (`explainer.py`)
Builds an evidence-only prompt and calls an injected LLM function. Falls back to a template
explanation if no LLM is configured or the call fails.

### CLI (`main.py`)
Reads files, calls the modules above and prints text or JSON.

## Data flow

Resume/JD files -> Parser -> Analyzer (skills, score) + Matcher (similarity) -> text / JSON output

With `--semantic`:

Resume -> Chunker -> Embedder -> Vector index
JD -> Chunker -> requirements -> query the index -> evidence + coverage -> (optional) Explainer
