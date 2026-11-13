# Market analysis service: FastAPI + LangGraph agent + three-model inference + policy retrieval.
# Build context is the repository root (electric/):
#   docker compose -f docker-compose.market.yml build
# Data, reranker weights and secrets are mounted/injected at run time, not baked in.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False \
    ELLECTRIC_RERANKER_PATH=/models/bge-reranker-v2-m3 \
    ELLECTRIC_OLLAMA_URL=http://localhost:11434 \
    ELLECTRIC_EMBEDDING_MODEL=bge-m3 \
    ELLECTRIC_LLM_THINKING=disabled

# libgomp1: OpenMP runtime for LightGBM/XGBoost; curl: container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ellectric/requirements-docker.lock /tmp/requirements-docker.lock
RUN pip install -r /tmp/requirements-docker.lock && rm /tmp/requirements-docker.lock

COPY ellectric /app/ellectric

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "ellectric.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
