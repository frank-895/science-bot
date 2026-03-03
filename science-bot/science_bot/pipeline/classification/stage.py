"""LLM-backed question-family classification stage."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from science_bot.pipeline.classification.schemas import (
    ClassificationStageInput,
    ClassificationStageOutput,
)
from science_bot.pipeline.contracts import QuestionFamily, parse_question_classification
from science_bot.providers import parse_structured

CLASSIFICATION_SYSTEM_PROMPT = """
You classify life-science analysis questions into one supported question family.

Supported families:
- aggregate: counting, means, medians, variances, skewness,
  percentages, proportions, ratios, and other summary statistics
- hypothesis_test: significance tests, p-values, correlations,
  distribution tests, and effect-size style questions framed as
  statistical tests
- regression: linear, logistic, ordinal logistic, polynomial,
  odds ratios, coefficients, predicted probabilities, and R-squared
- differential_expression: differential expression comparisons,
  significant genes, log fold changes, overlap across comparisons,
  and correction-method comparisons
- variant_filtering: variant counts, variant fractions/proportions,
  sample-specific variant counts, gene burden, and genomic filter
  questions

Rules:
- Choose exactly one supported family or unsupported.
- Do not extract operations, columns, filters, or execution details.
- Classify only from the wording of the question.
- Do not infer family from imagined dataset structure.
- If the question is ambiguous between families or outside scope, return unsupported.
- Include a non-empty reason only when family is unsupported.
""".strip()


class ClassificationResponse(BaseModel):
    """Structured LLM response for question-family classification."""

    model_config = ConfigDict(extra="forbid")

    family: QuestionFamily | Literal["unsupported"]
    reason: str | None = None


async def run_classification_stage(
    stage_input: ClassificationStageInput,
) -> ClassificationStageOutput:
    """Classify a natural-language question into a supported family.

    Args:
        stage_input: Classification stage input.

    Returns:
        ClassificationStageOutput: Supported or unsupported family classification.
    """
    response = await parse_structured(
        system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
        user_prompt=f"Question: {stage_input.question}",
        response_model=ClassificationResponse,
        trace_writer=stage_input.trace_writer,
        trace_stage="classification",
    )
    classification = parse_question_classification(
        response.model_dump(exclude_none=True)
    )
    return ClassificationStageOutput(classification=classification)
