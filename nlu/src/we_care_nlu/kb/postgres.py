"""Optional pgvector store. In-memory HybridRetriever remains the default for tests."""

from __future__ import annotations

from typing import Any

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS nlu_chunks (
  evidence_id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  document_title TEXT,
  section TEXT,
  page INT,
  english_text TEXT,
  amharic_text TEXT,
  review_status TEXT NOT NULL DEFAULT 'unreviewed',
  medical_subject TEXT,
  document_version TEXT,
  content_hash TEXT,
  embedding vector(%(dim)s)
);
CREATE INDEX IF NOT EXISTS nlu_chunks_embedding_idx
  ON nlu_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
"""


def connect(dsn: str):
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise RuntimeError("Install psycopg[binary] and pgvector for Postgres mode.") from exc
    conn = psycopg.connect(dsn)
    register_vector(conn)
    return conn


def init_schema(conn, dim: int = 768) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA.replace("%(dim)s", str(int(dim))))
    conn.commit()
