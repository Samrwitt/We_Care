# syntax=docker/dockerfile:1
# Default profile=span: chat sentence translator only (~smaller image).
# Full PDF pipeline: docker build --build-arg WE_CARE_MT_PROFILE=full .

FROM python:3.11-slim-bookworm

ARG WE_CARE_MT_PROFILE=span
# Default PyPI torch is the CUDA wheel (~2GB). The CPU index is ~200MB.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    WE_CARE_TORCH_THREADS=4

WORKDIR /app

# PDF rendering stack only for the full profile.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ "$WE_CARE_MT_PROFILE" = "full" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 fontconfig \
      && rm -rf /var/lib/apt/lists/*; \
    fi

COPY requirements.txt requirements-span.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install torch --index-url "${TORCH_INDEX_URL}" \
    && if [ "$WE_CARE_MT_PROFILE" = "full" ]; then \
         grep -vE '^[[:space:]]*torch([<>=!~ ].*)?$' requirements.txt > /tmp/reqs.txt; \
       else \
         grep -vE '^[[:space:]]*torch([<>=!~ ].*)?$' requirements-span.txt > /tmp/reqs.txt; \
       fi \
    && pip install -r /tmp/reqs.txt \
    && pip install --force-reinstall --no-deps torch --index-url "${TORCH_INDEX_URL}"

COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e .

COPY configs ./configs
COPY data ./data
COPY scripts ./scripts

RUN if [ "$WE_CARE_MT_PROFILE" = "full" ]; then \
      mkdir -p /usr/local/share/fonts/we-care \
      && cp -a data/fonts/. /usr/local/share/fonts/we-care/ \
      && fc-cache -f || true; \
    fi

VOLUME ["/app/data/source_pdfs", "/app/output", "/app/.hf_cache", "/app/cache", "/app/checkpoints"]

EXPOSE 8090
# 503 while NLLB loads still means the HTTP server is up (chat can show "loading").
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=30 \
  CMD ["python", "scripts/docker_health.py"]

ENTRYPOINT ["we-care-mt"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8090", "--model", "nllb_baseline_600m"]
