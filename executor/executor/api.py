"""Container-backed runtime for executing Python scripts."""

import asyncio
import importlib.metadata
import json
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_TIMEOUT_SECONDS = 30
_RUNS_ROOT = Path(".science-bot/runs")
_COMPOSE_FILE = Path("docker-compose.yml")
_SERVICE_NAME = "runner"
_MAX_OUTPUT_BYTES = 8192
_PYTHON_BINARY = "python"


class _ExecutorUnavailableError(RuntimeError):
    """Raised when execution workers are not ready."""


@dataclass(slots=True)
class _ExecutorState:
    """Mutable runtime state for worker scheduling and in-process queueing."""

    next_worker_index: int = 1
    index_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    worker_locks: dict[int, asyncio.Lock] = field(default_factory=dict)
    semaphore: asyncio.Semaphore | None = None
    semaphore_capacity: int = 0
    semaphore_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_STATE = _ExecutorState()


async def run_python(
    script: str,
    *,
    timeout_seconds: int | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    """Execute Python in a scaled worker and return a normalized result payload.

    Args:
        script: Python source code to execute.
        timeout_seconds: Optional timeout override.
        run_id: Optional grouping identifier for run artifacts.

    Returns:
        dict[str, object]: Structured execution result.

    Raises:
        RuntimeError: If workers are unavailable or execution infrastructure fails.
        ValueError: If script text is empty.
    """
    stripped_script = script.strip()
    if not stripped_script:
        raise ValueError("script must be non-empty.")

    effective_timeout_seconds = (
        timeout_seconds if timeout_seconds is not None else _DEFAULT_TIMEOUT_SECONDS
    )
    if effective_timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero.")

    worker_count = await asyncio.to_thread(_ensure_ready_and_discover_worker_count)
    semaphore = await _acquire_queue_slot(worker_count)

    try:
        worker_index = await _select_worker_index(worker_count)
        worker_lock = _worker_lock(worker_index)
        async with worker_lock:
            return await _run_with_worker(
                script=stripped_script,
                timeout_seconds=effective_timeout_seconds,
                run_id=run_id,
                worker_index=worker_index,
            )
    finally:
        semaphore.release()


def ensure_ready() -> None:
    """Validate Docker and worker readiness for Python execution.

    Raises:
        RuntimeError: If Docker or worker containers are unavailable.
    """
    _ensure_ready_and_discover_worker_count()


def packages_available() -> list[str]:
    """Return package names available in executor workers.

    Returns:
        list[str]: Sorted package names declared by executor requirements.
    """
    requirements = importlib.metadata.requires("executor") or []
    package_names = {_requirement_name(requirement) for requirement in requirements}
    return sorted(name for name in package_names if name is not None)


async def _acquire_queue_slot(worker_count: int) -> asyncio.Semaphore:
    """Acquire one in-process queue slot bounded by available workers.

    Args:
        worker_count: Number of healthy running workers.

    Returns:
        asyncio.Semaphore: Active semaphore used for queue release.
    """
    async with _STATE.semaphore_lock:
        if _STATE.semaphore is None or _STATE.semaphore_capacity != worker_count:
            _STATE.semaphore = asyncio.Semaphore(worker_count)
            _STATE.semaphore_capacity = worker_count
        semaphore = _STATE.semaphore

    await semaphore.acquire()
    return semaphore


async def _select_worker_index(worker_count: int) -> int:
    """Select a worker index via round-robin scheduling.

    Args:
        worker_count: Number of healthy running workers.

    Returns:
        int: Worker index in the range `[1, worker_count]`.
    """
    async with _STATE.index_lock:
        if _STATE.next_worker_index > worker_count:
            _STATE.next_worker_index = 1
        selected = _STATE.next_worker_index
        _STATE.next_worker_index += 1
        if _STATE.next_worker_index > worker_count:
            _STATE.next_worker_index = 1
        return selected


def _worker_lock(worker_index: int) -> asyncio.Lock:
    """Return the lock associated with one worker index.

    Args:
        worker_index: Worker index identifier.

    Returns:
        asyncio.Lock: Lock serialized per worker.
    """
    existing = _STATE.worker_locks.get(worker_index)
    if existing is not None:
        return existing

    created = asyncio.Lock()
    _STATE.worker_locks[worker_index] = created
    return created


async def _run_with_worker(
    *,
    script: str,
    timeout_seconds: int,
    run_id: str | None,
    worker_index: int,
) -> dict[str, object]:
    """Execute one script on a selected worker.

    Args:
        script: Python source code.
        timeout_seconds: Timeout in seconds.
        run_id: Optional run grouping identifier.
        worker_index: Worker index selected for execution.

    Returns:
        dict[str, object]: Structured execution result payload.
    """
    artifact_root = _RUNS_ROOT.resolve()
    effective_run_id = run_id or uuid.uuid4().hex
    attempt_id = uuid.uuid4().hex
    attempt_directory = artifact_root / effective_run_id / f"attempt_{attempt_id}"
    attempt_directory.mkdir(parents=True, exist_ok=True)

    host_script_path = attempt_directory / "attempt.py"
    host_script_path.write_text(script, encoding="utf-8")

    container_script_path = f"/runs/{effective_run_id}/attempt_{attempt_id}/attempt.py"
    command = [
        "docker",
        "compose",
        "-f",
        str(_COMPOSE_FILE),
        "exec",
        "-T",
        "--index",
        str(worker_index),
        _SERVICE_NAME,
        _PYTHON_BINARY,
        "-m",
        "executor._runner",
        "--script",
        container_script_path,
        "--timeout-seconds",
        str(timeout_seconds),
    ]

    loop = asyncio.get_running_loop()
    started = loop.time()
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds + 2
        )
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()

    duration_ms = int((loop.time() - started) * 1000)
    stdout_tail = _tail_bytes(stdout_bytes, _MAX_OUTPUT_BYTES)
    stderr_tail = _tail_bytes(stderr_bytes, _MAX_OUTPUT_BYTES)

    worker_label = f"{_SERVICE_NAME}[{worker_index}]"
    if timed_out:
        return {
            "status": "timeout",
            "answer": None,
            "error_type": "timeout",
            "error_message": "Execution exceeded executor timeout.",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "duration_ms": duration_ms,
            "worker": worker_label,
        }

    return _parse_runner_stdout(
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        duration_ms=duration_ms,
        worker_label=worker_label,
    )


def _ensure_ready_and_discover_worker_count() -> int:
    """Validate environment and return worker count.

    Returns:
        int: Number of healthy workers.

    Raises:
        RuntimeError: If Docker or workers are unavailable.
    """
    docker_result = _run_sync_command(["docker", "info"], raise_on_error=False)
    if docker_result.returncode != 0:
        raise _ExecutorUnavailableError(
            "Docker is unavailable. Start Docker Desktop or Docker Engine first."
        )

    worker_count = _discover_worker_count()
    if worker_count <= 0:
        raise _ExecutorUnavailableError(
            "Runner workers are not ready. Start them with: "
            f"docker compose -f {_COMPOSE_FILE} up -d --scale {_SERVICE_NAME}=1"
        )

    return worker_count


def _discover_worker_count() -> int:
    """Discover the number of healthy running workers.

    Returns:
        int: Number of workers in running and healthy state.
    """
    result = _run_sync_command(
        [
            "docker",
            "compose",
            "-f",
            str(_COMPOSE_FILE),
            "ps",
            "--format",
            "json",
            _SERVICE_NAME,
        ],
        raise_on_error=False,
    )
    if result.returncode != 0:
        return 0

    rows = _parse_compose_ps_json(result.stdout)
    healthy_rows = [
        row
        for row in rows
        if row.get("State") == "running" and row.get("Health") in {"", "healthy"}
    ]
    return len(healthy_rows)


def _run_sync_command(
    command: list[str],
    raise_on_error: bool,
) -> subprocess.CompletedProcess[str]:
    """Run one synchronous command for readiness checks.

    Args:
        command: Command with arguments.
        raise_on_error: Whether failures should raise.

    Returns:
        subprocess.CompletedProcess[str]: Completed process output.
    """
    return subprocess.run(
        command,
        check=raise_on_error,
        capture_output=True,
        text=True,
    )


def _parse_compose_ps_json(raw: str) -> list[dict[str, str]]:
    """Parse `docker compose ps --format json` output.

    Args:
        raw: Raw command output.

    Returns:
        list[dict[str, str]]: Parsed service rows.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []

    rows: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        state = item.get("State")
        health = item.get("Health")
        if isinstance(state, str) and isinstance(health, str):
            rows.append({"State": state, "Health": health})
    return rows


def _tail_bytes(value: bytes, max_bytes: int) -> str:
    """Decode and truncate bytes into a stable text tail.

    Args:
        value: Raw command output bytes.
        max_bytes: Maximum bytes retained.

    Returns:
        str: UTF-8 decoded tail.
    """
    return value[-max_bytes:].decode("utf-8", errors="replace")


def _parse_runner_stdout(
    *,
    stdout_tail: str,
    stderr_tail: str,
    duration_ms: int,
    worker_label: str,
) -> dict[str, object]:
    """Parse runner stdout JSON and return normalized execution results.

    Args:
        stdout_tail: Truncated runner stdout.
        stderr_tail: Truncated runner stderr.
        duration_ms: Execution duration in milliseconds.
        worker_label: Worker label used for execution.

    Returns:
        dict[str, object]: Normalized execution result payload.
    """
    try:
        payload = json.loads(stdout_tail)
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid_output",
            "answer": None,
            "error_type": "invalid_output",
            "error_message": str(exc),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "duration_ms": duration_ms,
            "worker": worker_label,
        }

    if not isinstance(payload, dict):
        return {
            "status": "invalid_output",
            "answer": None,
            "error_type": "invalid_output",
            "error_message": "Runner output must be a JSON object.",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "duration_ms": duration_ms,
            "worker": worker_label,
        }

    return {
        "status": payload.get("status", "invalid_output"),
        "answer": payload.get("answer"),
        "error_type": payload.get("error_type"),
        "error_message": payload.get("error_message"),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "duration_ms": duration_ms,
        "worker": worker_label,
    }


def _requirement_name(requirement: str) -> str | None:
    """Extract normalized package name from a requirement string.

    Args:
        requirement: Raw requirement string from package metadata.

    Returns:
        str | None: Normalized package name when parsing succeeds.
    """
    if not requirement:
        return None

    head = requirement.split(";", maxsplit=1)[0].strip()
    if not head:
        return None

    split_tokens = ("<", ">", "=", "!", "~", "[", " ")
    name = head
    for token in split_tokens:
        name = name.split(token, maxsplit=1)[0]
    normalized = name.strip()
    return normalized or None
