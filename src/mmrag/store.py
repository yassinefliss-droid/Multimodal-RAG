"""Stockage PostgreSQL + pgvector.

Tables :
- chunks       : morceaux de texte, tableaux en Markdown et descriptions VLM, avec
                 un vecteur (HNSW) et un tsvector (recherche plein texte).
- pages        : une ligne par page, avec un vecteur moyen pour le pré-filtrage.
- page_tokens  : les vecteurs multiples ColQwen2 de chaque page, pour le MaxSim.
- judgments    : verdicts humains saisis dans la web app.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .models import Element

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id   text PRIMARY KEY,
    title    text,
    sha256   text,
    n_pages  int
);

CREATE TABLE IF NOT EXISTS chunks (
    id         bigserial PRIMARY KEY,
    doc_id     text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page       int  NOT NULL,
    kind       text NOT NULL,            -- text | table | figure
    source     text NOT NULL,            -- docling | simple | vlm_description
    content    text NOT NULL,
    heading    text,
    bbox       jsonb,
    image_path text,
    embedding  vector({text_dim}) NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id, source);

CREATE TABLE IF NOT EXISTS pages (
    doc_id     text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page       int  NOT NULL,
    image_path text NOT NULL,
    pooled     vector({page_dim}),
    PRIMARY KEY (doc_id, page)
);
CREATE INDEX IF NOT EXISTS pages_pooled_idx ON pages USING hnsw (pooled vector_cosine_ops);

CREATE TABLE IF NOT EXISTS page_tokens (
    doc_id text NOT NULL,
    page   int  NOT NULL,
    idx    int  NOT NULL,
    emb    vector({page_dim}) NOT NULL,
    PRIMARY KEY (doc_id, page, idx),
    FOREIGN KEY (doc_id, page) REFERENCES pages(doc_id, page) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS judgments (
    id          bigserial PRIMARY KEY,
    created_at  timestamptz DEFAULT now(),
    question_id text,
    question    text NOT NULL,
    approach    text NOT NULL,
    answer      text NOT NULL,
    verdict     text NOT NULL CHECK (verdict IN ('correct', 'partial', 'wrong')),
    note        text
);
"""


class Store:
    def __init__(self, url: str, text_dim: int, page_dim: int):
        self.url = url
        self.text_dim = text_dim
        self.page_dim = page_dim

    @contextmanager
    def conn(self):
        with psycopg.connect(self.url, autocommit=False) as c:
            register_vector(c)
            yield c

    def init(self) -> None:
        with psycopg.connect(self.url, autocommit=True) as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        with self.conn() as c:
            c.execute(SCHEMA.format(text_dim=self.text_dim, page_dim=self.page_dim))

    # ------------------------------------------------------------------ écriture
    def upsert_document(self, doc_id: str, title: str | None, sha256: str | None, n_pages: int) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT INTO documents (doc_id, title, sha256, n_pages) VALUES (%s, %s, %s, %s)
                   ON CONFLICT (doc_id) DO UPDATE SET title = EXCLUDED.title,
                       sha256 = EXCLUDED.sha256, n_pages = EXCLUDED.n_pages""",
                (doc_id, title, sha256, n_pages),
            )

    def replace_chunks(self, doc_id: str, sources: list[str], elements: list[Element], vectors: np.ndarray) -> None:
        """Remplace les morceaux d'un document pour les sources données (ré-indexation idempotente)."""
        with self.conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_id = %s AND source = ANY(%s)", (doc_id, sources))
            with c.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (doc_id, page, kind, source, content, heading, bbox, image_path, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (e.doc_id, e.page, e.kind, e.source, e.content, e.heading,
                         json.dumps(e.bbox) if e.bbox else None, e.image_path, np.asarray(v, dtype=np.float32))
                        for e, v in zip(elements, vectors)
                    ],
                )

    def replace_page(self, doc_id: str, page: int, image_path: str, tokens: np.ndarray) -> None:
        tokens = np.asarray(tokens, dtype=np.float32)
        pooled = tokens.mean(axis=0)
        pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
        with self.conn() as c:
            c.execute("DELETE FROM pages WHERE doc_id = %s AND page = %s", (doc_id, page))
            c.execute("INSERT INTO pages (doc_id, page, image_path, pooled) VALUES (%s, %s, %s, %s)",
                      (doc_id, page, image_path, pooled))
            with c.cursor() as cur, cur.copy(
                "COPY page_tokens (doc_id, page, idx, emb) FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
                copy.set_types(["text", "int4", "int4", "vector"])
                for i, t in enumerate(tokens):
                    copy.write_row((doc_id, page, i, t))

    def save_judgment(self, question_id: str | None, question: str, approach: str, answer: str,
                      verdict: str, note: str | None) -> int:
        with self.conn() as c:
            row = c.execute(
                """INSERT INTO judgments (question_id, question, approach, answer, verdict, note)
                   VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                (question_id, question, approach, answer, verdict, note),
            ).fetchone()
        return row[0]

    # ------------------------------------------------------------------ lecture
    def search_vector(self, qvec: np.ndarray, sources: list[str], k: int) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                """SELECT id, doc_id, page, kind, source, content, image_path, 1 - (embedding <=> %s) AS score
                   FROM chunks WHERE source = ANY(%s) ORDER BY embedding <=> %s LIMIT %s""",
                (np.asarray(qvec, dtype=np.float32), sources, np.asarray(qvec, dtype=np.float32), k),
            ).fetchall()
        return [_row(r) for r in rows]

    def search_fulltext(self, query: str, sources: list[str], k: int) -> list[dict]:
        terms = sorted(set(re.findall(r"[a-z0-9]+", query.lower())))
        if not terms:
            return []
        tsq = " | ".join(terms)  # OU : une longue question ne doit pas exiger tous les mots
        with self.conn() as c:
            rows = c.execute(
                """SELECT id, doc_id, page, kind, source, content, image_path,
                          ts_rank_cd(tsv, to_tsquery('english', %s)) AS score
                   FROM chunks WHERE source = ANY(%s) AND tsv @@ to_tsquery('english', %s)
                   ORDER BY score DESC LIMIT %s""",
                (tsq, sources, tsq, k),
            ).fetchall()
        return [_row(r) for r in rows]

    def search_pages_maxsim(self, qtokens: np.ndarray, k: int, prefilter: int) -> list[dict]:
        """MaxSim ColPali calculé dans Postgres.

        score(q, page) = somme sur les tokens de la requête du max des produits
        scalaires avec les tokens de la page. Le pré-filtrage par vecteur moyen
        (HNSW) limite le calcul à `prefilter` pages ; prefilter <= 0 = toutes les pages.
        """
        qtokens = np.asarray(qtokens, dtype=np.float32)
        pooled = qtokens.mean(axis=0)
        pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
        cand = (
            "SELECT doc_id, page FROM pages ORDER BY pooled <=> %(pooled)s LIMIT %(n)s"
            if prefilter > 0 else "SELECT doc_id, page FROM pages"
        )
        sql = f"""
            WITH q AS (SELECT ord AS qi, emb FROM unnest(%(q)s::vector[]) WITH ORDINALITY AS t(emb, ord)),
                 cand AS ({cand})
            SELECT s.doc_id, s.page, p.image_path, sum(s.m) AS score
            FROM (
                SELECT pt.doc_id, pt.page, q.qi, max(-(pt.emb <#> q.emb)) AS m
                FROM page_tokens pt JOIN cand USING (doc_id, page) CROSS JOIN q
                GROUP BY pt.doc_id, pt.page, q.qi
            ) s JOIN pages p USING (doc_id, page)
            GROUP BY s.doc_id, s.page, p.image_path
            ORDER BY score DESC LIMIT %(k)s"""
        with self.conn() as c:
            rows = c.execute(sql, {"q": list(qtokens), "pooled": pooled, "n": prefilter, "k": k}).fetchall()
        return [{"doc_id": r[0], "page": r[1], "image_path": r[2], "score": float(r[3])} for r in rows]

    def page_image(self, doc_id: str, page: int) -> str | None:
        with self.conn() as c:
            r = c.execute("SELECT image_path FROM pages WHERE doc_id = %s AND page = %s", (doc_id, page)).fetchone()
        return r[0] if r else None

    def stats(self) -> dict:
        with self.conn() as c:
            docs = c.execute("SELECT count(*) FROM documents").fetchone()[0]
            by_source = dict(c.execute("SELECT source || '/' || kind, count(*) FROM chunks GROUP BY 1").fetchall())
            pages = c.execute("SELECT count(*) FROM pages").fetchone()[0]
            toks = c.execute("SELECT count(*) FROM page_tokens").fetchone()[0]
        return {"documents": docs, "chunks": by_source, "pages": pages, "page_tokens": toks}

    def judgments(self) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT question_id, question, approach, verdict, note, created_at FROM judgments ORDER BY id"
            ).fetchall()
        keys = ["question_id", "question", "approach", "verdict", "note", "created_at"]
        return [dict(zip(keys, r)) for r in rows]


def _row(r) -> dict:
    return {"id": r[0], "doc_id": r[1], "page": r[2], "kind": r[3], "source": r[4],
            "content": r[5], "image_path": r[6], "score": float(r[7])}


def default_store(settings) -> Store:
    return Store(settings.database_url, settings.text_dim, settings.page_dim)


def image_exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()
