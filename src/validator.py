from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import MCQItem, ValidationResult


def _validate_base_schema(mcq: MCQItem) -> tuple[list[str], dict[str, bool]]:
    errors: list[str] = []
    checks: dict[str, bool] = {}

    required_ok = bool(
        mcq.question_statement.strip()
        and mcq.correct_explanation.strip()
        and mcq.source_reference.document_title.strip()
        and mcq.source_reference.page_number
        and mcq.source_reference.section_or_topic.strip()
    )
    checks["Required fields & non-empty strings"] = required_ok
    if not mcq.question_statement.strip():
        errors.append("Question statement cannot be empty.")
    if not mcq.correct_explanation.strip():
        errors.append("Correct explanation cannot be empty.")
    if not mcq.source_reference.document_title.strip():
        errors.append("Source reference document title cannot be empty.")
    if not mcq.source_reference.section_or_topic.strip():
        errors.append("Source reference section/topic cannot be empty.")

    labels = [opt.label for opt in mcq.options]
    four_options_ok = len(mcq.options) == 4
    label_ok = sorted(labels) == ["A", "B", "C", "D"]
    option_texts = [opt.text.strip().lower() for opt in mcq.options]
    distinct_ok = len(set(option_texts)) == 4 and all(option_texts)
    checks["Exactly 4 options A-D"] = four_options_ok and label_ok
    checks["Distinct, non-empty options"] = distinct_ok
    if not four_options_ok:
        errors.append("Exactly 4 options are required.")
    if not label_ok:
        errors.append("Options must be labeled A, B, C, D exactly once.")
    if not distinct_ok:
        errors.append("All 4 option texts must be distinct and non-empty.")

    valid_labels = {"A", "B", "C", "D"}
    answer_ok = bool(mcq.correct_answers) and set(mcq.correct_answers).issubset(valid_labels)
    checks["Correct answers use valid labels"] = answer_ok
    if not answer_ok:
        errors.append("Correct answers must only contain valid labels A, B, C, D.")

    if mcq.question_type == "Single Correct":
        count_ok = len(mcq.correct_answers) == 1
    else:
        count_ok = len(mcq.correct_answers) >= 2
    checks["Question type matches answer count"] = count_ok
    if not count_ok:
        if mcq.question_type == "Single Correct":
            errors.append("Single Correct questions must have exactly one correct answer.")
        else:
            errors.append("Multiple Correct questions must have at least two correct answers.")

    selected = set(mcq.correct_answers)
    incorrect_labels = valid_labels - selected
    provided = set(mcq.distractor_explanations.keys())
    missing = sorted(list(incorrect_labels - provided))
    extra = sorted(list(provided & selected))
    explanation_ok = bool(mcq.correct_explanation.strip()) and not missing and not extra
    checks["Explanation and distractor consistency"] = explanation_ok
    if missing:
        errors.append(f"Distractor explanations are missing for incorrect option(s): {missing}.")
    if extra:
        errors.append(f"Distractor explanations must not include correct option(s): {extra}.")

    return errors, checks


def detect_duplicate_pairs(items: list[MCQItem], threshold: float = 0.75) -> list[dict[str, Any]]:
    if len(items) < 2:
        return []

    try:
        statements = [item.question_statement for item in items]
        matrix = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(statements))
    except ValueError:
        return []

    pairs: list[dict[str, Any]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            score = float(matrix[i, j])
            if score > threshold:
                pairs.append(
                    {
                        "first_id": items[i].id,
                        "second_id": items[j].id,
                        "similarity_percent": round(score * 100, 2),
                        "threshold_percent": round(threshold * 100, 2),
                    }
                )
    return pairs


def detect_duplicates(items: list[MCQItem], threshold: float = 0.75) -> dict[str, list[str]]:
    pairs = detect_duplicate_pairs(items, threshold)
    duplicate_flags: dict[str, list[str]] = {}
    for pair in pairs:
        first = pair["first_id"]
        second = pair["second_id"]
        detail = (
            f"Potential duplicate with {second} "
            f"(similarity={pair['similarity_percent']:.2f}%)."
        )
        duplicate_flags.setdefault(first, []).append(detail)
        duplicate_flags.setdefault(second, []).append(
            f"Potential duplicate with {first} "
            f"(similarity={pair['similarity_percent']:.2f}%)."
        )
    return duplicate_flags


def batch_uniqueness(items: list[MCQItem]) -> float:
    pairs = detect_duplicate_pairs(items)
    if len(items) < 2:
        return 100.0
    # Average complement of stem similarity across every pair.
    try:
        statements = [item.question_statement for item in items]
        matrix = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(statements))
        scores = [float(matrix[i, j]) for i in range(len(items)) for j in range(i + 1, len(items))]
        return round(max(0.0, (1.0 - sum(scores) / len(scores)) * 100), 2)
    except ValueError:
        return 100.0


def validate_mcq_item(
    mcq: MCQItem,
    duplicate_flags: list[str] | None = None,
    duplicate_pairs: list[dict[str, Any]] | None = None,
    uniqueness_score: float = 100.0,
) -> ValidationResult:
    errors, checks = _validate_base_schema(mcq)
    warnings: list[str] = []

    if len(mcq.question_statement.split()) < 8:
        warnings.append("Question statement is very short; consider adding context.")
    if len(mcq.correct_explanation.split()) < 10:
        warnings.append("Correct explanation is brief; consider improving rationale depth.")

    return ValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        duplicate_flags=duplicate_flags or [],
        duplicate_pairs=duplicate_pairs or [],
        batch_uniqueness_score=uniqueness_score,
        check_results=checks,
    )


def validate_batch(items: list[MCQItem]) -> dict[str, ValidationResult]:
    duplicate_map = detect_duplicates(items)
    duplicate_pairs = detect_duplicate_pairs(items)
    uniqueness_score = batch_uniqueness(items)
    results: dict[str, ValidationResult] = {}

    for mcq in items:
        results[mcq.id] = validate_mcq_item(
            mcq,
            duplicate_flags=duplicate_map.get(mcq.id, []),
            duplicate_pairs=[pair for pair in duplicate_pairs if mcq.id in (pair["first_id"], pair["second_id"])],
            uniqueness_score=uniqueness_score,
        )

    return results


def validate_drafts(raw_items: list[dict[str, Any]]) -> dict[str, ValidationResult]:
    parsed_items: list[MCQItem] = []
    raw_errors: dict[str, ValidationResult] = {}

    for draft in raw_items:
        item_id = str(draft.get("id", "unknown"))
        try:
            parsed_items.append(MCQItem.model_validate(draft))
        except Exception as exc:
            raw_errors[item_id] = ValidationResult(
                is_valid=False,
                errors=[f"Schema validation failed: {exc}"],
                warnings=[],
                duplicate_flags=[],
            )

    parsed_results = validate_batch(parsed_items)
    parsed_results.update(raw_errors)
    return parsed_results
