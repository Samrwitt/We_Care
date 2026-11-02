# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    WE_CARE_TORCH_THREADS=4

WORKDIR /app

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt

# CPU torch first, then sentence-transformers; re-pin CPU torch so ST cannot leave a CUDA wheel.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url "${TORCH_INDEX_URL}" \
    && pip install "sentence-transformers>=3.0.0" \
    && pip install --force-reinstall --no-deps torch --index-url "${TORCH_INDEX_URL}"

COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e .

COPY configs ./configs
COPY data ./data
COPY scripts ./scripts

EXPOSE 8088
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=20 \
  CMD ["python", "scripts/docker_health.py"]

# One worker: embedding + index state is in-process and not share-safe across workers.
CMD ["uvicorn", "we_care_nlu.api.app:app", "--host", "0.0.0.0", "--port", "8088", "--workers", "1", "--timeout-keep-alive", "30"]
