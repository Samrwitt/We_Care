# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# Wheels exist for every dependency, so the image does not install a compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Default PyPI torch is the CUDA wheel (~2GB). The CPU index is ~200MB.
# GPU rebuild: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install torch --index-url "${TORCH_INDEX_URL}" \
    && grep -vE '^[[:space:]]*torch([<>=!~ ].*)?$' requirements.txt > /tmp/requirements.notorch.txt \
    && pip install -r /tmp/requirements.notorch.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e .

COPY configs ./configs
COPY data ./data
COPY scripts ./scripts

COPY data/fonts /usr/local/share/fonts/we-care
RUN fc-cache -f || true

VOLUME ["/app/data/source_pdfs", "/app/output", "/app/.hf_cache", "/app/cache", "/app/checkpoints"]

ENTRYPOINT ["we-care-mt"]
CMD ["--help"]
