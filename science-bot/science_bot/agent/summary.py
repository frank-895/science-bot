"""Utilities for compact step-history summaries."""

from science_bot.agent.contracts import AgentStepRecord


def summarize_steps(steps: list[AgentStepRecord], max_chars: int = 2400) -> str:
    """Summarize prior agent steps into a bounded text payload.

    Args:
        steps: Historical step records.
        max_chars: Maximum summary length.

    Returns:
        str: Compact summary text for prompting.
    """
    if not steps:
        return "No prior steps."

    lines: list[str] = []
    for step in steps[-8:]:
        fields = [f"iter={step.iteration}", f"decision={step.decision}"]
        if step.execution_status:
            fields.append(f"exec_status={step.execution_status}")
        if step.execution_error:
            fields.append(f"exec_error={step.execution_error[:120]}")
        if step.answer:
            fields.append(f"answer={step.answer[:120]}")
        if step.reason:
            fields.append(f"reason={step.reason[:120]}")
        lines.append("; ".join(fields))

    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    return summary[-max_chars:]
