FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY data /app/data
COPY tests /app/tests
COPY scripts /app/scripts
RUN pip install --upgrade pip && pip install -e ".[dev]"
EXPOSE 8088
CMD ["uvicorn", "we_care_nlu.api.app:app", "--host", "0.0.0.0", "--port", "8088"]
