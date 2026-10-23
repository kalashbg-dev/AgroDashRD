# One image for the web app and the jobs: `agrodash web` by default, or any other
# command (`agrodash migrate`, `agrodash ingest --drive`, `agrodash forecast`).

FROM ghcr.io/astral-sh/uv:0.12.10@sha256:2bb3ebca0a796a155094a27773d290c4b074572e6107f171d88d086682fd2500 AS uv

FROM python:3.12.15-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: they change less often than the code, so this layer stays cached.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12.15-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70
RUN useradd --system --uid 10001 --no-create-home agrodash
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
# Streamlit reads its theme and limits from .streamlit/ in the working directory.
COPY .streamlit ./.streamlit
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    AGRODASH_ENV=production \
    NUMBA_CACHE_DIR=/tmp/numba
USER agrodash
EXPOSE 8501
ENTRYPOINT ["agrodash"]
CMD ["web", "--port", "8501"]
