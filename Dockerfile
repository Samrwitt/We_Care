FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        fontconfig \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install pytest

# CPU torch wheel is sufficient inside the default image.
# For GPU, rebuild FROM nvidia/cuda and install torch with CUDA.

COPY data/fonts /usr/local/share/fonts/we-care
RUN fc-cache -f || true

VOLUME ["/app/data/source_pdfs", "/app/output", "/app/.hf_cache", "/app/cache", "/app/checkpoints"]

ENTRYPOINT ["we-care-mt"]
CMD ["--help"]
