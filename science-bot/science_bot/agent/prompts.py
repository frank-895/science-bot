"""Prompt builders for the iterative agent runtime."""

from pathlib import Path

DECISION_SCHEMA_TEXT = """Return JSON matching exactly one of:
1) {"decision":"run_python","script":"<python>"}
2) {"decision":"respond","answer":"<final answer>"}
3) {"decision":"need_info","reason":"<missing information>"}
Do not add extra keys.
"""


def build_system_prompt(max_iterations: int) -> str:
    """Build the system prompt for one decision turn.

    Args:
        max_iterations: Maximum allowed iterations for this run.

    Returns:
        str: System prompt text.
    """
    return (
        "You are a life-science data analysis agent.\n"
        "You must answer the question by using short, targeted Python scripts.\n"
        f"You have at most {max_iterations} iterations total.\n"
        "Start executing quickly, avoid broad exploration, and provide a candidate "
        "answer as soon as evidence is sufficient.\n"
        "Prefer deterministic scripts and concise outputs.\n"
        f"{DECISION_SCHEMA_TEXT}"
    )


def build_user_prompt(
    *,
    question: str,
    capsule_path: Path,
    available_packages: list[str],
    step_summary: str,
    iteration: int,
    max_iterations: int,
) -> str:
    """Build the user prompt for one decision turn.

    Args:
        question: Natural language question.
        capsule_path: Capsule filesystem path.
        available_packages: Packages available for execution.
        step_summary: Compact summary of prior steps.
        iteration: Current 1-based iteration.
        max_iterations: Maximum iteration budget.

    Returns:
        str: User prompt text.
    """
    remaining = max_iterations - iteration + 1
    package_list = ", ".join(available_packages) if available_packages else "(none)"
    return (
        f"Question:\n{question}\n\n"
        f"Capsule path:\n{capsule_path}\n\n"
        f"Available packages:\n{package_list}\n\n"
        f"Iteration:\n{iteration}/{max_iterations} (remaining={remaining})\n\n"
        "Step summary:\n"
        f"{step_summary}\n\n"
        "Respond with one valid JSON object only."
    )
