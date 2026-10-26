# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt

# LaBSE retrieval. Install CPU torch last so sentence-transformers cannot swap in the CUDA wheel.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "sentence-transformers>=3.0.0" \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu --upgrade

COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e .

COPY configs ./configs
COPY data ./data
COPY scripts ./scripts

EXPOSE 8088
CMD ["uvicorn", "we_care_nlu.api.app:app", "--host", "0.0.0.0", "--port", "8088"]
