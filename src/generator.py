from __future__ import annotations

import json
import os
import random
import time
import uuid
from typing import Any

from dotenv import load_dotenv

from src.schemas import MCQBatch

try:
    from google import genai
except ImportError:  # pragma: no cover - environment-dependent import
    genai = None

load_dotenv()


class MCQGenerator:
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
                raise RuntimeError("Live mode requires GEMINI_API_KEY. Add it to .env or enable Mock Mode for a demo.")
            if genai is None:
                raise RuntimeError("Live mode requires the google-genai package. Enable Mock Mode for a demo or install google-genai.")
            self.client = genai.Client(api_key=self.api_key)
        elif genai is None:
            self.mock_mode = True

    @staticmethod
    def _build_model_candidates(primary_model: str) -> list[str]:
        candidates = [primary_model, "gemini-2.5-flash-lite"]
        unique_candidates: list[str] = []
        for model in candidates:
            if model not in unique_candidates:
                unique_candidates.append(model)
        return unique_candidates

    @staticmethod
    def _difficulty_guidance(difficulty: str) -> str:
        mapping = {
            "Easy": "Focus on conceptual recall, definitions, and direct identification.",
            "Medium": "Test statutory application, exceptions, and moderate practical interpretation.",
            "Hard": "Use multi-provision case scenarios, edge cases, and tricky exception handling.",
        }
        return mapping.get(difficulty, mapping["Medium"])

    def _build_prompt(
        self,
        context_chunks: list[dict[str, Any]],
        subject: str,
        chapter: str,
        topic: str,
        difficulty: str,
        question_type: str,
        num_questions: int,
    ) -> str:
        context_blocks: list[str] = []
        for idx, chunk in enumerate(context_chunks, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"[Context {idx}]",
                        f"Subject: {chunk.get('subject', '')}",
                        f"Chapter: {chunk.get('chapter', '')}",
                        f"Topic: {chunk.get('topic', '')}",
                        f"Document: {chunk.get('document_title', '')}",
                        f"Source URL: {chunk.get('source_url', '')}",
                        f"Page: {chunk.get('page_number', '')}",
                        f"Text: {chunk.get('text', '')}",
                    ]
                )
            )

        return f"""
You are an expert CA Final assessment designer.
Generate exactly {num_questions} high-quality MCQs in strict JSON schema output.

Target:
- Subject: {subject}
- Chapter: {chapter or 'General'}
- Topic: {topic or chapter or 'General'}
- Difficulty: {difficulty}
- Question Type: {question_type}

Difficulty Guidance:
- {self._difficulty_guidance(difficulty)}

Hard Rules:
1. Use only facts from the provided context blocks.
2. Do not hallucinate section numbers, limits, rates, case laws, or statutory thresholds.
3. Each question must include exact source attribution with document title and page number.
4. Create exactly 4 options labeled A, B, C, D.
5. For Single Correct, provide exactly 1 correct answer label.
6. For Multiple Correct, provide at least 2 correct answer labels.
7. Explanations must explicitly tie back to context excerpts.
8. Ensure distractors are plausible but clearly incorrect based on context.

Return only schema-compliant JSON.

Context:
{chr(10).join(context_blocks)}
""".strip()

    @staticmethod
    def _parse_model_response(response: Any) -> MCQBatch:
        if hasattr(response, "parsed") and response.parsed is not None:
            parsed = response.parsed
            if isinstance(parsed, MCQBatch):
                return parsed
            return MCQBatch.model_validate(parsed)

        if hasattr(response, "text") and response.text:
            return MCQBatch.model_validate(json.loads(response.text))

        if hasattr(response, "candidates") and response.candidates:
            content = response.candidates[0].content.parts[0].text
            return MCQBatch.model_validate(json.loads(content))

        raise ValueError("Unable to parse Gemini response into MCQBatch.")

    def _generate_with_gemini(
        self,
        context_chunks: list[dict[str, Any]],
        subject: str,
        chapter: str,
        topic: str,
        difficulty: str,
        question_type: str,
        num_questions: int,
    ) -> MCQBatch:
        if self.client is None:
            raise RuntimeError("Gemini client is not initialized.")

        prompt = self._build_prompt(
            context_chunks=context_chunks,
            subject=subject,
            chapter=chapter,
            topic=topic,
            difficulty=difficulty,
            question_type=question_type,
            num_questions=num_questions,
        )

        last_error: Exception | None = None
        attempts_log: list[str] = []

        for model_name in self.model_candidates:
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={
                            "temperature": 0.2,
                            "response_mime_type": "application/json",
                            "response_schema": MCQBatch,
                        },
                    )
                    return self._parse_model_response(response)
                except Exception as exc:  # pragma: no cover - API/runtime dependent
                    last_error = exc
                    attempts_log.append(f"{model_name} attempt {attempt + 1}: {exc}")
                    if attempt < 2:
                        error_text = str(exc).lower()
                        backoff_seconds = 2**attempt
                        if "429" in error_text or "rate" in error_text:
                            backoff_seconds += 1
                        time.sleep(backoff_seconds)

        joined_attempts = " | ".join(attempts_log)
        error_reason = str(last_error) if last_error is not None else "Unknown live API failure"
        raise RuntimeError(
            "Live MCQ generation failed after retries across models "
            f"({', '.join(self.model_candidates)}). "
            f"Reason: {error_reason}. Attempts: {joined_attempts}"
        )

    def _generate_mock(
        self,
        context_chunks: list[dict[str, Any]],
        subject: str,
        chapter: str,
        topic: str,
        difficulty: str,
        question_type: str,
        num_questions: int,
    ) -> MCQBatch:
        if not context_chunks:
            context_chunks = [
                {
                    "subject": subject,
                    "chapter": chapter or "General",
                    "topic": topic or chapter or "General",
                    "document_title": "Seeded ICAI Context",
                    "source_url": "https://boslive.icai.org/",
                    "page_number": 1,
                    "text": "Supply, ITC, and Capital Gains concepts are evaluated using context-grounded facts.",
                }
            ]

        questions: list[dict[str, Any]] = []

        for index in range(num_questions):
            context = context_chunks[index % len(context_chunks)]
            context_text = str(context.get("text", ""))
            excerpt = context_text[:280].strip()
            cue = str(context.get("topic") or topic or chapter or "Core Topic")

            options = {
                "A": f"Statement A on {cue} aligned with the statutory context.",
                "B": f"Statement B on {cue} that overextends the rule beyond context.",
                "C": f"Statement C on {cue} that omits a key condition.",
                "D": f"Statement D on {cue} that confuses scope and exception.",
            }

            if question_type == "Single Correct":
                correct_answers = ["A"]
            else:
                correct_answers = ["A", random.choice(["C", "D"])]

            incorrect_labels = [label for label in ["A", "B", "C", "D"] if label not in correct_answers]
            distractor_explanations = {
                label: "This option is inconsistent with the source excerpt or misses statutory conditions."
                for label in incorrect_labels
            }

            questions.append(
                {
                    "id": f"mcq_{uuid.uuid4().hex[:10]}",
                    "subject": str(context.get("subject", subject)),
                    "chapter": str(context.get("chapter", chapter or "General")),
                    "topic": str(context.get("topic", topic or chapter or "General")),
                    "difficulty": difficulty,
                    "question_type": question_type,
                    "question_statement": (
                        f"Based on ICAI CA Final context on {cue}, identify the most accurate statement "
                        f"for {difficulty} level assessment."
                    ),
                    "options": [{"label": label, "text": text} for label, text in options.items()],
                    "correct_answers": correct_answers,
                    "correct_explanation": (
                        "The selected answer(s) directly match conditions stated in the provided ICAI excerpt, "
                        "including scope, trigger, and exception boundaries."
                    ),
                    "distractor_explanations": distractor_explanations,
                    "source_reference": {
                        "document_title": str(context.get("document_title", "Seeded ICAI Context")),
                        "source_url": str(context.get("source_url", "https://boslive.icai.org/")),
                        "page_number": int(context.get("page_number", 1)),
                        "section_or_topic": cue,
                        "excerpt": excerpt or "Context excerpt unavailable.",
                    },
                    "status": "Draft",
                }
            )

        return MCQBatch.model_validate({"questions": questions})

    def generate_mcqs(
        self,
        context_chunks: list[dict[str, Any]],
        subject: str,
        chapter: str,
        topic: str,
        difficulty: str,
        question_type: str,
        num_questions: int,
    ) -> MCQBatch:
        if self.mock_mode:
            return self._generate_mock(
                context_chunks=context_chunks,
                subject=subject,
                chapter=chapter,
                topic=topic,
                difficulty=difficulty,
                question_type=question_type,
                num_questions=num_questions,
            )

        return self._generate_with_gemini(
            context_chunks=context_chunks,
            subject=subject,
            chapter=chapter,
            topic=topic,
            difficulty=difficulty,
            question_type=question_type,
            num_questions=num_questions,
        )
