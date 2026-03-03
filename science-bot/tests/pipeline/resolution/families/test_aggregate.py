import pytest
from science_bot.pipeline.resolution.families.aggregate import (
    AggregateResolutionDecision,
    AggregateResolvedPlan,
    build_aggregate_plan_from_decision,
)


def test_build_aggregate_plan_from_decision():
    plan = build_aggregate_plan_from_decision(
        AggregateResolutionDecision(
            action="finalize",
            reason="done",
            filename="data.csv",
            operation="mean",
            value_column="value",
            return_format="number",
        ),
        require_text=lambda value, _name: value,
        require_value=lambda value, _name: value,
    )

    assert plan.filename == "data.csv"


def test_aggregate_resolved_plan_requires_value_column():
    with pytest.raises(ValueError):
        AggregateResolvedPlan(
            filename="data.csv",
            operation="mean",
            value_column=None,
            filters=[],
            numerator_filters=[],
            denominator_filters=[],
            return_format="number",
        )
