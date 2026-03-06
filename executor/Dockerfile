FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir uv==0.9.2

WORKDIR /executor
COPY executor/pyproject.toml /executor/pyproject.toml
COPY executor/uv.lock /executor/uv.lock
COPY executor /executor
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 runner
USER runner

ENV PATH="/executor/.venv/bin:${PATH}"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -m executor._runner --help >/dev/null 2>&1 || exit 1

CMD ["tail", "-f", "/dev/null"]
