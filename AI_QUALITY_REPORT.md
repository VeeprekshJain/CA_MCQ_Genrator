# AI Quality Report

Generated: 2026-11-23

## 1) Evaluation Methodology

The project evaluates MCQ quality through a three-layer pipeline:

1. Deterministic validation (rule-based, reproducible)
2. LLM-as-a-Judge scoring (structured rubric scoring)
3. Human-in-the-loop educator review (domain assurance)

Live generation and judging use:
- Primary: `gemini-2.5-flash`
- Fallback: `gemini-2.5-flash-lite`
- Up to 3 attempts per model, with exponential backoff for 429/rate-limit, transient API, and malformed JSON failures.

Rubric criteria:
1. Relevance
2. Factual Grounding
3. Clarity
4. Difficulty Alignment
5. Distractor Quality
6. Explanation Consistency

## 2) Review Matrix

| Layer | Automated checks | UI output | Publication decision |
|---|---|---|---|
| Deterministic | Required fields; four distinct A-D options; valid answers; type/count consistency; distractor coverage; TF-IDF duplicate detection and uniqueness | Green/warning checklist, exact pair similarity, batch pass rate | Never publish structurally failed drafts |
| LLM-as-a-Judge | Six 1–5 rubric dimensions and actionable critique | Scores table and low-score warnings | Assist educator prioritization; not final authority |
| Human Educator | Amendments, ambiguity, exam fairness, statutory nuance | Draft / Approved / Needs Revision | Required for final publication |

## 3) Issues Found and Resolved During Testing

### Issue 1: Obsolete Gemini model names
Obsolete `gemini-3.*` names were present in generation/evaluation code and documentation. Updated both live classes to primary `gemini-2.5-flash` and fallback `gemini-2.5-flash-lite`.

### Issue 2: Silent live mock fallback
The live path could previously switch to mock behavior when the SDK was missing or return live failures unclearly. Live mode now raises descriptive errors and the UI explains API-key, quota/rate-limit, network, and Mock Mode remedies.

### Issue 3: Incomplete deterministic checklist details
Validation has been upgraded to report per-rule pass/fail details, exact near-duplicate similarity percentages, and a batch uniqueness/diversity percentage.

### Issue 4: UI review flow was not walkthrough-ready
The UI now starts with a pipeline architecture banner and review matrix, sidebar captions, a batch summary strip, source-attribution pills, per-rule validation cards, editable answer explanations, status controls, and a 6-criterion quality view.

### Issue 5: Export or review could hide structural failures
Refresh-time validation now gates the draft review display and clearly identifies missing fields, invalid labels, cardinality mismatches, distractor omissions, and duplicate stems.

## 4) Known Limitations

1. Retrieval remains lexical-hybrid (BM25 + TF-IDF), so deep semantic recall can differ from dense vector retrieval.
2. Rubric scores are model-dependent and require human statutory review.
3. TF-IDF duplicate detection can miss substantially reworded semantic duplicates.
4. ICAI amendments and effective dates must be checked against the latest study material before publication.

## 5) Recommendation

Use the system as an educator-assist drafting and screening pipeline. Deterministic rules should block malformed drafts, LLM judging should guide revision, and final statutory and exam-fairness approval must remain with a human educator.
