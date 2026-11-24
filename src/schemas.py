from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Score = Annotated[int, Field(ge=1, le=5)]
OptionLabel = Literal["A", "B", "C", "D"]
DifficultyLevel = Literal["Easy", "Medium", "Hard"]
QuestionType = Literal["Single Correct", "Multiple Correct"]
MCQStatus = Literal["Draft", "Approved", "Needs Revision"]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MCQOption(StrictBaseModel):
    label: OptionLabel
    text: Annotated[str, Field(min_length=1)]


class SourceReference(StrictBaseModel):
    document_title: Annotated[str, Field(min_length=1)]
    source_url: Annotated[str, Field(min_length=1)]
    page_number: Annotated[int, Field(ge=1)]
    section_or_topic: Annotated[str, Field(min_length=1)]
    excerpt: Annotated[str, Field(min_length=1)]


class MCQItem(StrictBaseModel):
    id: Annotated[str, Field(min_length=1)]
    subject: Annotated[str, Field(min_length=1)]
    chapter: Annotated[str, Field(min_length=1)]
    topic: Annotated[str, Field(min_length=1)]
    difficulty: DifficultyLevel
    question_type: QuestionType
    question_statement: Annotated[str, Field(min_length=1)]
    options: list[MCQOption]
    correct_answers: list[OptionLabel]
    correct_explanation: Annotated[str, Field(min_length=1)]
    distractor_explanations: dict[OptionLabel, Annotated[str, Field(min_length=1)]]
    source_reference: SourceReference
    status: MCQStatus = "Draft"

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[MCQOption]) -> list[MCQOption]:
        if len(value) != 4:
            raise ValueError("Exactly 4 options are required.")
        labels = [opt.label for opt in value]
        if sorted(labels) != ["A", "B", "C", "D"]:
            raise ValueError("Options must use labels A, B, C, D exactly once.")
        option_texts = [opt.text.strip().lower() for opt in value]
        if len(set(option_texts)) != 4:
            raise ValueError("Option text must be unique across all 4 options.")
        return value

    @field_validator("correct_answers")
    @classmethod
    def validate_correct_answers(cls, value: list[OptionLabel]) -> list[OptionLabel]:
        if not value:
            raise ValueError("At least one correct answer is required.")
        if len(set(value)) != len(value):
            raise ValueError("Correct answers must not contain duplicates.")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> "MCQItem":
        allowed_labels = {"A", "B", "C", "D"}
        selected = set(self.correct_answers)
        if not selected.issubset(allowed_labels):
            raise ValueError("Correct answer labels must be within A, B, C, D.")

        if self.question_type == "Single Correct" and len(selected) != 1:
            raise ValueError("Single Correct questions must have exactly 1 correct answer.")
        if self.question_type == "Multiple Correct" and len(selected) < 2:
            raise ValueError("Multiple Correct questions must have at least 2 correct answers.")

        incorrect_labels = allowed_labels - selected
        provided_labels = set(self.distractor_explanations.keys())
        if not incorrect_labels.issubset(provided_labels):
            missing = sorted(list(incorrect_labels - provided_labels))
            raise ValueError(
                f"Distractor explanations are missing for incorrect options: {missing}."
            )

        return self


class MCQBatch(StrictBaseModel):
    questions: list[MCQItem]

    @field_validator("questions")
    @classmethod
    def non_empty_batch(cls, value: list[MCQItem]) -> list[MCQItem]:
        if not value:
            raise ValueError("Batch must contain at least one question.")
        return value


class ValidationResult(StrictBaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duplicate_flags: list[str] = Field(default_factory=list)
    check_results: dict[str, bool] = Field(default_factory=dict)
    duplicate_pairs: list[dict[str, Any]] = Field(default_factory=list)
    batch_uniqueness_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Batch uniqueness percentage derived from maximum pairwise TF-IDF cosine similarity",
    )


class QualityScore(StrictBaseModel):
    relevance: Score
    factual_grounding: Score
    clarity: Score
    difficulty_alignment: Score
    distractor_quality: Score
    explanation_consistency: Score
    overall_score: Annotated[float, Field(ge=1.0, le=5.0)]
    critique: Annotated[str, Field(min_length=1)]

