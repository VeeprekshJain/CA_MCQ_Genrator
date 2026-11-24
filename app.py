from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.evaluator import QualityEvaluator, export_quality_report
from src.generator import MCQGenerator
from src.ingest import (
    CHUNKS_PATH,
    RAW_DATA_DIR,
    ensure_knowledge_base,
    ingest_pdf_file,
    ingest_raw_pdfs,
    load_chunks,
    save_chunks,
    seed_sample_chunks,
)
from src.retriever import HybridRetriever
from src.schemas import MCQItem
from src.validator import validate_drafts

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
QUALITY_REPORT_PATH = ROOT_DIR / "AI_QUALITY_REPORT.md"

st.set_page_config(page_title="CA Final AI MCQ Generator", layout="wide")
st.title("AI-Powered CA Final MCQ Generation System")
st.caption("Context-grounded MCQ drafting for IDT and DT with deterministic validation and quality scoring.")

with st.expander("How This AI Pipeline Works", expanded=False):
    st.markdown(
        "**1. ICAI Context Retrieval (BM25 + TF-IDF)** → "
        "**2. Structured Gemini Generation (Pydantic)** → "
        "**3. Deterministic Guardrails** → "
        "**4. LLM-as-a-Judge & Human Educator Review**"
    )
    st.markdown("The stages are intentionally separated: automation is fast and auditable, while statutory sign-off remains human-led.")
    st.table(
        {
            "Review Layer": ["Deterministic", "LLM-as-a-Judge", "Human Educator"],
            "What it checks": [
                "Required fields, option labels/text, answer-cardinality, distractor coverage, source attribution, and near-duplicates",
                "Relevance, factual grounding, clarity, difficulty fit, distractor quality, and explanation consistency",
                "Exam fairness, statutory currency, ambiguity, amendments, and final publication approval",
            ],
            "Output": ["PASS/FAIL errors and checklist", "1–5 rubric scores and actionable critique", "Draft / Approved / Needs Revision"],
        }
    )


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "retrieved_chunks": [],
        "mcq_drafts": [],
        "validation_results": {},
        "quality_scores": {},
        "last_generation_params": {},
        "mock_mode": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_retriever() -> HybridRetriever:
    chunks = ensure_knowledge_base(seed_sample=True)
    return HybridRetriever(chunks)


def parse_valid_items(drafts: list[dict[str, Any]]) -> list[MCQItem]:
    items: list[MCQItem] = []
    for draft in drafts:
        try:
            items.append(MCQItem.model_validate(draft))
        except Exception:
            continue
    return items


def refresh_validation() -> None:
    st.session_state["validation_results"] = validate_drafts(st.session_state["mcq_drafts"])


def generate_mcqs(
    mock_mode: bool,
    subject: str,
    chapter: str,
    topic: str,
    num_questions: int,
    difficulty: str,
    question_type: str,
) -> None:
    retriever = build_retriever()
    query = " ".join(part for part in [chapter, topic, subject] if part).strip() or subject
    retrieved = retriever.retrieve(
        query=query,
        subject=subject,
        chapter=chapter or None,
        top_k=max(4, min(10, num_questions + 2)),
    )

    if not retrieved:
        st.warning("No matching context chunks found for this filter. Try removing chapter filter.")
        return

    generator = MCQGenerator(mock_mode=mock_mode)
    batch = generator.generate_mcqs(
        context_chunks=retrieved,
        subject=subject,
        chapter=chapter or "General",
        topic=topic or chapter or "General",
        difficulty=difficulty,
        question_type=question_type,
        num_questions=num_questions,
    )

    st.session_state["retrieved_chunks"] = retrieved
    st.session_state["mcq_drafts"] = [question.model_dump() for question in batch.questions]
    st.session_state["quality_scores"] = {}
    st.session_state["last_generation_params"] = {
        "subject": subject,
        "chapter": chapter,
        "topic": topic,
        "difficulty": difficulty,
        "question_type": question_type,
        "num_questions": num_questions,
    }
    refresh_validation()


def render_sidebar() -> None:
    st.sidebar.header("Generation Controls")

    has_api_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
    if has_api_key:
        st.sidebar.success("API Key: configured")
    else:
        st.sidebar.warning("API Key: not found (Mock Mode recommended)")

    mock_mode = st.sidebar.toggle("Mock Mode", value=st.session_state.get("mock_mode", not has_api_key))
    st.session_state["mock_mode"] = mock_mode

    subject = st.sidebar.selectbox(
        "Subject",
        ["Indirect Tax Laws (IDT)", "Direct Tax Laws (DT)"],
        index=0,
    )
    st.sidebar.caption("IDT is mandatory for the Final tax paper; DT is included as a bonus/related demo track.")
    chapter = st.sidebar.text_input("Chapter", value="Supply under GST" if "IDT" in subject else "Capital Gains")
    st.sidebar.caption("Choose the exact ICAI chapter and optionally narrow it with a topic.")
    topic = st.sidebar.text_input("Topic", value="")
    st.sidebar.caption("Topic is optional; it narrows retrieval toward a provision, exception, or case type.")
    num_questions = st.sidebar.slider("Number of Questions", min_value=1, max_value=10, value=3)
    difficulty = st.sidebar.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    st.sidebar.caption("Easy = recall; Medium = statutory application; Hard = multi-provision case study.")
    question_type = st.sidebar.selectbox("Question Type", ["Single Correct", "Multiple Correct"], index=0)
    st.sidebar.caption("Single Correct requires 1 answer; Multiple Correct requires at least 2 answers.")

    if st.sidebar.button("Generate Draft MCQs", type="primary", use_container_width=True):
        with st.spinner("Generating grounded MCQ drafts..."):
            try:
                generate_mcqs(
                    mock_mode=mock_mode,
                    subject=subject,
                    chapter=chapter.strip(),
                    topic=topic.strip(),
                    num_questions=num_questions,
                    difficulty=difficulty,
                    question_type=question_type,
                )
            except Exception as exc:
                st.error(
                    "MCQ generation failed. Check the GEMINI_API_KEY, API quota/rate limits, model availability, "
                    "and network access. You can also enable Mock Mode for an offline demo walkthrough.\n\n"
                    f"Technical detail: {exc}"
                )


init_session_state()
ensure_knowledge_base(seed_sample=True)
render_sidebar()


tab_generate, tab_quality, tab_kb = st.tabs(
    ["Generate & Review Drafts", "Quality Evaluation", "Knowledge Base & Ingestion"]
)

with tab_generate:
    st.subheader("Retrieved Context Transparency")
    chunks = st.session_state.get("retrieved_chunks", [])
    if not chunks:
        st.info("Generate MCQs to see retrieved context chunks.")
    else:
        for idx, chunk in enumerate(chunks, start=1):
            with st.expander(
                f"Context {idx}: {chunk.get('document_title', 'Unknown')} | Page {chunk.get('page_number', 'N/A')}"
            ):
                st.write(f"Subject: {chunk.get('subject', '')}")
                st.write(f"Chapter: {chunk.get('chapter', '')}")
                st.write(f"Source URL: {chunk.get('source_url', '')}")
                st.write(chunk.get("text", ""))

    st.subheader("Editable MCQ Draft Cards")
    drafts = st.session_state.get("mcq_drafts", [])
    if not drafts:
        st.info("No drafts yet. Use sidebar controls and click Generate Draft MCQs.")
    else:
        for idx, draft in enumerate(drafts):
            draft_id = str(draft.get("id", f"draft_{idx + 1}"))
            header = f"Q{idx + 1} | {draft.get('topic', 'General')} | {draft.get('difficulty', 'Medium')}"
            with st.expander(header, expanded=(idx == 0)):
                draft["question_statement"] = st.text_area(
                    "Question Statement",
                    value=draft.get("question_statement", ""),
                    key=f"q_stmt_{draft_id}",
                )

                option_map = {opt["label"]: opt["text"] for opt in draft.get("options", []) if "label" in opt}
                updated_options: list[dict[str, str]] = []
                for label in ["A", "B", "C", "D"]:
                    value = st.text_input(
                        f"Option {label}",
                        value=option_map.get(label, ""),
                        key=f"q_opt_{draft_id}_{label}",
                    )
                    updated_options.append({"label": label, "text": value})
                draft["options"] = updated_options

                draft["correct_answers"] = st.multiselect(
                    "Correct Answer(s)",
                    options=["A", "B", "C", "D"],
                    default=draft.get("correct_answers", []),
                    key=f"q_correct_{draft_id}",
                )

                draft["correct_explanation"] = st.text_area(
                    "Correct Explanation",
                    value=draft.get("correct_explanation", ""),
                    key=f"q_cexp_{draft_id}",
                )

                current_distractors = draft.get("distractor_explanations", {})
                updated_distractors: dict[str, str] = {}
                for label in ["A", "B", "C", "D"]:
                    if label not in draft["correct_answers"]:
                        updated_distractors[label] = st.text_input(
                            f"Why option {label} is incorrect",
                            value=current_distractors.get(label, ""),
                            key=f"q_dexp_{draft_id}_{label}",
                        )
                draft["distractor_explanations"] = updated_distractors

                draft["status"] = st.selectbox(
                    "Status",
                    options=["Draft", "Approved", "Needs Revision"],
                    index=["Draft", "Approved", "Needs Revision"].index(draft.get("status", "Draft")),
                    key=f"q_status_{draft_id}",
                )

                st.session_state["mcq_drafts"][idx] = draft

        refresh_validation()
        results = st.session_state.get("validation_results", {})
        total = len(drafts)
        passing = sum(
            1 for draft in drafts
            if (results.get(str(draft.get("id", ""))).is_valid
                if results.get(str(draft.get("id", ""))) else False)
        )
        pass_rate = round(passing / total * 100, 1) if total else 0.0
        uniqueness = next(
            (r.batch_uniqueness_score for r in results.values()),
            100.0,
        )
        approved = sum(1 for draft in drafts if draft.get("status", "Draft") == "Approved")

        st.markdown("### Batch Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Generated", total)
        c2.metric("Deterministic Pass Rate", f"{pass_rate:.1f}%")
        c3.metric("Uniqueness / Avg Diversity", f"{uniqueness:.2f}%")
        c4.metric("Approved Count", approved)

        for idx, draft in enumerate(drafts):
            draft_id = str(draft.get("id", f"draft_{idx + 1}"))
            header = f"Q{idx + 1} | {draft.get('topic', 'General')} | {draft.get('difficulty', 'Medium')}"
            with st.expander(header, expanded=(idx == 0)):
                source = draft.get("source_reference", {})
                st.markdown(
                    f"**Source Attribution:** {source.get('document_title', 'Unknown document')}  |  "
                    f"Page {source.get('page_number', 'N/A')}  |  "
                    f"Section/Topic: {source.get('section_or_topic', 'N/A')}"
                )
                st.caption(f"Source URL: {source.get('source_url', 'Not provided')}")

                st.markdown("**Deterministic Validation Checklist**")
                result = results.get(draft_id)
                if result:
                    for rule, passed in result.check_results.items():
                        st.write(f"{'✅' if passed else '⚠️'} {rule}")
                    for duplicate in result.duplicate_flags:
                        st.warning(duplicate)
                    if result.errors:
                        st.error("\n".join(f"• {error}" for error in result.errors))
                else:
                    st.warning("Validation details unavailable.")

                st.markdown("**Editable Draft**")
                draft["question_statement"] = st.text_area(
                    "Question Statement", value=draft.get("question_statement", ""), key=f"q_stmt_{draft_id}"
                )

                option_map = {opt["label"]: opt["text"] for opt in draft.get("options", []) if "label" in opt}
                updated_options: list[dict[str, str]] = []
                for label in ["A", "B", "C", "D"]:
                    value = st.text_input(
                        f"Option {label}", value=option_map.get(label, ""), key=f"q_opt_{draft_id}_{label}"
                    )
                    updated_options.append({"label": label, "text": value})
                draft["options"] = updated_options

                draft["correct_answers"] = st.multiselect(
                    "Correct Answer(s)", ["A", "B", "C", "D"],
                    default=draft.get("correct_answers", []), key=f"q_correct_{draft_id}"
                )
                draft["correct_explanation"] = st.text_area(
                    "Correct Explanation", value=draft.get("correct_explanation", ""), key=f"q_cexp_{draft_id}"
                )

                current_distractors = draft.get("distractor_explanations", {})
                updated_distractors: dict[str, str] = {}
                for label in ["A", "B", "C", "D"]:
                    if label not in draft["correct_answers"]:
                        updated_distractors[label] = st.text_input(
                            f"Why option {label} is incorrect",
                            value=current_distractors.get(label, ""),
                            key=f"q_dexp_{draft_id}_{label}",
                        )
                draft["distractor_explanations"] = updated_distractors

                st.markdown("**Educator Review Status**")
                draft["status"] = st.selectbox(
                    "Status", ["Draft", "Approved", "Needs Revision"],
                    index=["Draft", "Approved", "Needs Revision"].index(draft.get("status", "Draft")),
                    key=f"q_status_{draft_id}",
                )
                st.session_state["mcq_drafts"][idx] = draft

        refresh_validation()
        st.markdown("### Export")
        export_mode = st.radio("Export Mode", ["All Drafts", "Approved Only"], horizontal=True)
        export_payload = st.session_state["mcq_drafts"]
        if export_mode == "Approved Only":
            export_payload = [item for item in export_payload if item.get("status") == "Approved"]
        st.download_button(
            "Download MCQs as JSON", json.dumps(export_payload, indent=2, ensure_ascii=False),
            file_name="mcq_export.json", mime="application/json"
        )

with tab_quality:
    st.subheader("LLM-as-a-Judge 6-Criteria Rubric")
    st.table(
        {
            "Criterion": ["Relevance", "Factual Grounding", "Clarity", "Difficulty Alignment", "Distractor Quality", "Explanation Consistency"],
            "What 1–5 scores mean": [
                "Does the stem match the selected chapter/topic?",
                "Are facts and answer rationale supported by retrieved ICAI context?",
                "Is the question unambiguous and professionally worded?",
                "Does complexity match Easy, Medium, or Hard?",
                "Are wrong options plausible, distinct, and clearly incorrect?",
                "Do explanations match the answer key and source excerpt?",
            ],
        }
    )

    refresh_validation()
    validation = st.session_state.get("validation_results", {})
    uniqueness = next((r.batch_uniqueness_score for r in validation.values()), 100.0)
    st.metric("Deterministic Uniqueness & Diversity", f"{uniqueness:.2f}%")
    st.caption("Computed from average pairwise TF-IDF cosine complement: 100% means no measured stem overlap in the batch.")

    if st.button("Run Quality Evaluation", type="primary"):
        drafts = st.session_state.get("mcq_drafts", [])
        valid_items = parse_valid_items(drafts)
        if not valid_items:
            st.warning("No schema-valid MCQs are available. Correct the deterministic errors in the draft cards first.")
        else:
            with st.spinner("Running the live/mock LLM judge..."):
                try:
                    evaluator = QualityEvaluator(mock_mode=bool(st.session_state.get("mock_mode", True)))
                    scores = evaluator.evaluate_batch(valid_items, st.session_state.get("retrieved_chunks", []))
                    st.session_state["quality_scores"] = {k: v.model_dump() for k, v in scores.items()}
                    export_quality_report(scores, QUALITY_REPORT_PATH)
                except Exception as exc:
                    st.error(
                        "Quality evaluation failed. Check the GEMINI_API_KEY, quota/rate limits, model availability, "
                        "and network access; otherwise enable Mock Mode for a demo.\n\n"
                        f"Technical detail: {exc}"
                    )

    quality_scores = st.session_state.get("quality_scores", {})
    if not quality_scores:
        st.info("No quality scores yet. Generate drafts, then run the 6-criteria evaluation.")
    else:
        rows = [
            {
                "MCQ ID": mcq_id,
                "Relevance": score["relevance"],
                "Factual Grounding": score["factual_grounding"],
                "Clarity": score["clarity"],
                "Difficulty": score["difficulty_alignment"],
                "Distractors": score["distractor_quality"],
                "Explanation": score["explanation_consistency"],
                "Overall": score["overall_score"],
            }
            for mcq_id, score in quality_scores.items()
        ]
        st.dataframe(rows, use_container_width=True)

        st.markdown("### Plain-English Actionable Critiques")
        for mcq_id, score in quality_scores.items():
            with st.expander(f"{mcq_id} | Overall {score['overall_score']:.2f}/5"):
                st.write(score["critique"])
                low = [
                    name for name, key in {
                        "Relevance": "relevance", "Factual grounding": "factual_grounding",
                        "Clarity": "clarity", "Difficulty alignment": "difficulty_alignment",
                        "Distractor quality": "distractor_quality", "Explanation consistency": "explanation_consistency",
                    }.items() if score[key] <= 3
                ]
                if low:
                    st.warning("Prioritize review in: " + ", ".join(low) + ".")
                else:
                    st.success("No criterion is below 4/5; educator review is still required before publication.")

        if QUALITY_REPORT_PATH.exists():
            st.success("A quality report has been exported to AI_QUALITY_REPORT.md.")

with tab_kb:
    st.subheader("Knowledge Base Status")
    chunks = load_chunks(CHUNKS_PATH)
    st.write(f"Indexed chunk count: {len(chunks)}")

    if chunks:
        summary: dict[tuple[str, str], int] = {}
        for chunk in chunks:
            key = (str(chunk.get("subject", "Unknown")), str(chunk.get("chapter", "Unknown")))
            summary[key] = summary.get(key, 0) + 1
        summary_rows = [
            {"subject": subject, "chapter": chapter, "chunk_count": count}
            for (subject, chapter), count in sorted(summary.items())
        ]
        st.dataframe(summary_rows, use_container_width=True)

    st.markdown("### Upload ICAI PDFs")
    upload_subject = st.selectbox(
        "Upload Subject",
        ["Indirect Tax Laws (IDT)", "Direct Tax Laws (DT)"],
        key="upload_subject",
    )
    upload_chapter = st.text_input("Upload Chapter", value="General", key="upload_chapter")
    upload_source_url = st.text_input("Upload Source URL", value="https://boslive.icai.org/", key="upload_source_url")
    upload_edition = st.text_input("Edition/Attempt", value="May/Nov 2026", key="upload_edition")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Ingest Uploaded PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            existing = load_chunks()
            all_new_chunks: list[dict[str, Any]] = []
            for file in uploaded_files:
                target_path = RAW_DATA_DIR / file.name
                target_path.write_bytes(file.getbuffer())
                all_new_chunks.extend(
                    ingest_pdf_file(
                        pdf_path=target_path,
                        subject=upload_subject,
                        chapter=upload_chapter,
                        source_url=upload_source_url,
                        edition_or_attempt=upload_edition,
                        topic=upload_chapter,
                    )
                )

            merged = {chunk["chunk_id"]: chunk for chunk in existing}
            for chunk in all_new_chunks:
                merged[chunk["chunk_id"]] = chunk
            save_chunks(list(merged.values()))
            st.success(f"Ingested {len(all_new_chunks)} chunks from uploaded files.")

    st.markdown("### Raw Folder Ingestion")
    if st.button("Run Ingestion on data/raw"):
        updated = ingest_raw_pdfs(
            default_subject=upload_subject,
            default_chapter=upload_chapter,
            default_source_url=upload_source_url,
            edition_or_attempt=upload_edition,
        )
        st.success(f"Ingestion complete. Total indexed chunks: {len(updated)}")

    st.markdown("### Seed Demo Data")
    if st.button("Seed Sample IDT/DT Chunks"):
        seed_sample_chunks(save=True)
        st.success("Sample chunks seeded successfully.")
