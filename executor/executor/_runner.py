"""Container-side runner that executes Python scripts and emits JSON output."""

import argparse
import json
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for runner invocation.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(prog="executor._runner")
    parser.add_argument("--script", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser


def main() -> int:
    """Execute one script and print a JSON result to stdout.

    Returns:
        int: Process exit code.
    """
    args = build_parser().parse_args()
    script_path = Path(args.script)

    try:
        completed = subprocess.run(
            ["python", str(script_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        print(
            json.dumps(
                {
                    "status": "timeout",
                    "answer": None,
                    "error_type": "timeout",
                    "error_message": str(exc),
                }
            )
        )
        return 124

    stdout_text = completed.stdout.strip()
    stderr_text = completed.stderr.strip()

    if completed.returncode == 0:
        print(
            json.dumps(
                {
                    "status": "completed",
                    "answer": stdout_text,
                    "error_type": None,
                    "error_message": None,
                }
            )
        )
        return 0

    print(
        json.dumps(
            {
                "status": "failed",
                "answer": None,
                "error_type": "runtime_error",
                "error_message": (
                    stderr_text or f"Script exited with {completed.returncode}."
                ),
            }
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
