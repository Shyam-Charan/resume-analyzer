# Architecture

## Components

### Parser Module
Extracts raw text from PDF and text resumes.

### Analyzer Module
Identifies skills, experience, and education using NLP.

### Matcher Module
Compares resume content against job descriptions.

### Output Layer
Returns match score and extracted information.

## Data Flow

Resume Input → Parser → Analyzer → Matcher → Score Output
