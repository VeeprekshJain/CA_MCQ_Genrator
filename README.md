# AI-Powered MCQ Generation System for CA Final

This project generates context-grounded CA Final MCQ drafts (IDT and DT), validates them deterministically, and evaluates them using an LLM-as-a-Judge rubric in a Streamlit interface.

## Local Setup and Run

1. Create and activate a virtual environment.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install project dependencies.

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_google_ai_studio_key
```

4. Start the Streamlit app.

```powershell
streamlit run app.py
```

Notes:
- If `GEMINI_API_KEY` is missing, keep Mock Mode ON for generation/evaluation.
- The app auto-seeds sample chunks if no processed knowledge base exists.

## Architecture Overview

### 1) Ingestion (`src/ingest.py`)
- Uses PyMuPDF to extract textbook text from PDFs in `data/raw/`.
- Builds overlapping chunks (default size/overlap settings preserve local context across boundaries).
- Stores chunk corpus and metadata in `data/processed/chunks.json` and `data/processed/sources_metadata.json`.

### 2) Retrieval (`src/retriever.py`)
- Hybrid ranking: BM25 + TF-IDF cosine with score fusion.
- Supports filters by subject/chapter and returns top-k chunks with citation fields.
- Retrieval metadata is displayed in UI for transparency.

### 3) Generation (`src/generator.py`)
- Uses Gemini with strict structured output (`response_schema=MCQBatch`).
- Prompt adapts by difficulty and question type while enforcing context grounding.
- Uses retries with exponential backoff; mock generation is available for offline/demo mode.

### 4) Deterministic Validation (`src/validator.py`)
- Enforces exactly 4 options (`A-D`), valid answer labels, answer-count consistency with question type, and distractor explanation coverage.
- Runs TF-IDF cosine duplicate detection to flag near-duplicate question stems.

### 5) LLM-as-a-Judge Evaluation (`src/evaluator.py`)
- Uses `gemini-2.5-flash` first, then falls back to `gemini-2.5-flash-lite`.
- Uses `google-genai` structured output with `response_schema=QualityScore`.
- Evaluates six rubric dimensions and computes `overall_score` + critique.

### 6) Streamlit UI (`app.py`)
- Generate/review tab: context preview, editable MCQs, validation messages, JSON export.
- Quality tab: run LLM judge, view rubric scores, export markdown report.
- Knowledge base tab: upload/ingest PDFs and inspect indexed chunk coverage.

## Model Selection Rationale

Primary model: `gemini-2.5-flash`

Why this model:
- Native structured JSON support with `response_schema` reduces post-processing failures.
- Low latency supports iterative classroom-authoring workflows.
- Suitable for grounded generation and nuanced 1â€“5 rubric evaluation.

Fallback model: `gemini-2.5-flash-lite`

Why fallback exists:
- Preserves service continuity during primary-model availability or quota issues.
- Retains structured output behavior and clear live-failure reporting.

## Prompt Design Strategy

Difficulty controls:
- Easy: direct conceptual recall and definition matching.
- Medium: moderate statutory application and exception checks.
- Hard: multi-condition interpretation and edge-case reasoning.

Question type controls:
- Single Correct: prompt enforces exactly one correct option.
- Multiple Correct: prompt enforces at least two correct options.

Grounding controls:
- Explicitly disallow unsupported legal claims and fabricated thresholds/sections.
- Require source attribution fields (`document_title`, `source_url`, `page_number`, excerpt linkage).
- Require concise explanation consistency between answer key and cited context.

## Context Grounding and Retrieval Trade-offs

Current approach:
- PyMuPDF page extraction.
- Overlapping chunking to reduce boundary information loss.
- Hybrid BM25 + TF-IDF retrieval for legal keyword precision and lightweight semantic matching.

Trade-offs versus dense vector databases:
- Pros: simpler stack, no vector index infrastructure, reproducible local behavior, lower operational overhead for small/medium corpora.
- Cons: weaker semantic recall for paraphrased content and long-range concept alignment compared with dense embeddings + ANN search.
- Practical implication: for assignment-scale corpora, hybrid lexical retrieval is sufficient and auditable; for production at larger scale, dense retrieval + re-ranking is the next upgrade path.

## ICAI Reference Dataset (Expected Source Inputs)

| Paper | Subject | ICAI Official URL | Chapter Coverage (example) | Exam Applicability |
|---|---|---|---|---|
| Final Paper 5 | Indirect Tax Laws (IDT) | https://www.icai.org/post/sm-final-paper5 | Supply under GST, Time/Place/Value of Supply, ITC, Returns, Assessments | May 2026 and Nov 2026 |
| Final Paper 4 | Direct Tax Laws (DT) | https://www.icai.org/post/sm-final-paper4 | Capital Gains, PGBP, Deductions, Residential Status, Assessment Procedures | May 2026 and Nov 2026 |

Copyright and repository policy:
- Full ICAI copyrighted PDFs are intentionally not committed.
- `.gitignore` excludes raw PDFs via `data/raw/*.pdf`.
- Only derived metadata/chunks needed for demonstration are kept in repository-safe form.

## Sample Reproducible Artifact

- A full sample input/output trace is provided at `samples/sample_run.json`.
- It includes realistic parameters, retrieved citations, generated Single/Multiple Correct MCQs, deterministic validation outcomes, and LLM quality scores.

