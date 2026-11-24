from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.schemas import MCQItem, QualityScore

try:
    from google import genai
except ImportError:  # pragma: no cover - environment-dependent import
    genai = None

load_dotenv()


class QualityEvaluator:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        mock_mode: bool | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if mock_mode is None:
            mock_mode = not bool(resolved_key)

        self.api_key = resolved_key
        self.model_name = model_name
        self.model_candidates = self._build_model_candidates(model_name)
        self.mock_mode = bool(mock_mode)
        self.client = None

        if not self.mock_mode:
            if not self.api_key:
                raise RuntimeError("Live evaluation requires GEMINI_API_KEY. Add it to .env or enable Mock Mode for a demo.")
            if genai is None:
                raise RuntimeError("Live evaluation requires the google-genai package. Enable Mock Mode for a demo or install google-genai.")
            self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _build_model_candidates(primary_model: str) -> list[str]:
        candidates = [primary_model, "gemini-2.5-flash-lite"]
        unique_candidates: list[str] = []
        for model in candidates:
            if model not in unique_candidates:
                unique_candidates.append(model)
        return unique_candidates

    def _build_prompt(self, item: MCQItem, source_chunks: list[dict[str, Any]]) -> str:
        context = "\n\n".join(
            [
                (
                    f"Document: {chunk.get('document_title', '')} | Page: {chunk.get('page_number', '')} | "
                    f"Chapter: {chunk.get('chapter', '')}\n"
                    f"Text: {chunk.get('text', '')}"
                )
                for chunk in source_chunks[:6]
            ]
        )

        return f"""
You are an LLM judge evaluating CA Final MCQ quality against supplied source context.
Score each dimension from 1 to 5 and provide a concise but specific critique.

Rubric dimensions:
- relevance
- factual_grounding
- clarity
- difficulty_alignment
- distractor_quality
- explanation_consistency

Question JSON:
{item.model_dump_json(indent=2)}

Source Context:
{context}

Rules:
1. Ground factual score only on source evidence.
2. Penalize hallucinated legal claims and unsupported section references.
3. Penalize if distractors are weak or ambiguous.
4. Keep critique action-oriented for an educator.

Return strict JSON matching the schema.
""".strip()

    @staticmethod
    def _parse_response(response: Any) -> QualityScore:
        if hasattr(response, "parsed") and response.parsed is not None:
            parsed = response.parsed
            if isinstance(parsed, QualityScore):
                return parsed
            return QualityScore.model_validate(parsed)

        if hasattr(response, "text") and response.text:
            return QualityScore.model_validate(json.loads(response.text))

        raise ValueError("Unable to parse evaluator response.")

    def _evaluate_with_gemini(self, item: MCQItem, source_chunks: list[dict[str, Any]]) -> QualityScore:
        if self.client is None:
            raise RuntimeError("Gemini client is not initialized.")

        prompt = self._build_prompt(item=item, source_chunks=source_chunks)

        last_error: Exception | None = None
        attempts_log: list[str] = []

        for model_name in self.model_candidates:
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={
                            "temperature": 0.1,
                            "response_mime_type": "application/json",
                            "response_schema": QualityScore,
                        },
                    )
                    return self._parse_response(response)
                except Exception as exc:  # pragma: no cover - API/runtime dependent
                    last_error = exc
                    attempts_log.append(f"{model_name} attempt {attempt + 1}: {exc}")
                    if attempt < 2:
                        time.sleep(2**attempt)

        joined_attempts = " | ".join(attempts_log)
        error_reason = str(last_error) if last_error is not None else "Unknown live API failure"
        raise RuntimeError(
            "Live quality evaluation failed after retries across models "
            f"({', '.join(self.model_candidates)}). "
            f"Reason: {error_reason}. Attempts: {joined_attempts}"
        )

    @staticmethod
    def _evaluate_mock(item: MCQItem) -> QualityScore:
        clarity = 5 if len(item.question_statement.split()) >= 12 else 3
        explanation_consistency = 5 if len(item.correct_explanation.split()) >= 15 else 3
        distractor_quality = 4 if len(item.distractor_explanations) >= 2 else 2
        relevance = 4
        factual_grounding = 4
        difficulty_alignment = 4

        scores = [
            relevance,
            factual_grounding,
            clarity,
            difficulty_alignment,
            distractor_quality,
            explanation_consistency,
        ]
        overall = round(sum(scores) / len(scores), 2)

        critique = (
            "Mock evaluation: question is generally usable for draft review. "
            "Re-check statutory precision, strengthen distractor nuance, and ensure explanations cite context language."
        )

        return QualityScore(
            relevance=relevance,
            factual_grounding=factual_grounding,
            clarity=clarity,
            difficulty_alignment=difficulty_alignment,
            distractor_quality=distractor_quality,
            explanation_consistency=explanation_consistency,
            overall_score=overall,
            critique=critique,
        )

    def evaluate_item(self, item: MCQItem, source_chunks: list[dict[str, Any]]) -> QualityScore:
        if self.mock_mode:
            return self._evaluate_mock(item)
        return self._evaluate_with_gemini(item=item, source_chunks=source_chunks)

    def evaluate_batch(
        self,
        items: list[MCQItem],
        source_chunks: list[dict[str, Any]],
    ) -> dict[str, QualityScore]:
        scores: dict[str, QualityScore] = {}
        for item in items:
            scores[item.id] = self.evaluate_item(item=item, source_chunks=source_chunks)
        return scores


def export_quality_report(scores: dict[str, QualityScore], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# AI Quality Evaluation Report",
        "",
        f"Generated: {timestamp}",
        "",
        "| MCQ ID | Relevance | Factual | Clarity | Difficulty | Distractors | Explanation | Overall | Critique |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    if not scores:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | No evaluations generated. |")

    for mcq_id, score in scores.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    mcq_id,
                    str(score.relevance),
                    str(score.factual_grounding),
                    str(score.clarity),
                    str(score.difficulty_alignment),
                    str(score.distractor_quality),
                    str(score.explanation_consistency),
                    f"{score.overall_score:.2f}",
                    score.critique.replace("|", "/"),
                ]
            )
            + " |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
