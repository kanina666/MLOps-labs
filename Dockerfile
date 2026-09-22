FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Set environment variables:
# - PYTHONDONTWRITEBYTECODE: prevent generating .pyc files at runtime
# - PYTHONUNBUFFERED: stream logs directly to stdout/stderr without buffering
# - UV_COMPILE_BYTECODE: pre-compile bytecode on install for faster startup
# - UV_LINK_MODE: copy files into venv instead of hardlinking across mounts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"


COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

RUN uv sync --frozen --no-dev

# Security best practice: run application as a non-privileged user
RUN groupadd --system appuser && useradd --system -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz').read()" || exit 1

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "mlops_labs.main:app", "--host", "0.0.0.0", "--port", "8000"]
