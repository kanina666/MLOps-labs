FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /bin/uv

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"


COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --group web --no-install-project

COPY src ./src

RUN uv sync --locked --no-dev --group web --no-editable

ARG APP_VERSION=""
ARG VCS_REF="local"
ENV APP_VERSION_OVERRIDE=${APP_VERSION}
LABEL org.opencontainers.image.title="mlops-labs" \
      org.opencontainers.image.version=${APP_VERSION} \
      org.opencontainers.image.revision=${VCS_REF}

RUN groupadd --system appuser && useradd --system -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()" || exit 1

CMD ["uvicorn", "mlops_labs.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
