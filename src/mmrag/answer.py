"""Génération de la réponse par le LLM vision, avec citations vérifiées.

Chaque preuve est numérotée [S1], [S2]... Le modèle doit citer ses sources ;
on vérifie ensuite que chaque citation renvoie bien à une preuve fournie,
et on la traduit en (document, page).
"""
from __future__ import annotations

import re
import time

from .models import Answer, Citation, Context, Retrieved
from .vlm import VLM, image_part, text_part

NOT_FOUND = "NOT_FOUND"

SYSTEM = f"""You answer questions about scientific papers using ONLY the provided sources.
Sources are numbered [S1], [S2]... and may be text passages, tables or images of PDF pages.
Rules:
- Answer concisely (1 to 3 sentences). Give exact numbers with their units when relevant.
- After each claim, cite the source(s) it comes from, like [S2].
- If the sources do not contain the answer, reply exactly {NOT_FOUND}.
- Never use outside knowledge."""

CITATION_RE = re.compile(r"\[S(\d+)\]")


def build_parts(question: str, contexts: list[Context]) -> list[dict]:
    parts: list[dict] = []
    for c in contexts:
        header = f"[{c.label}] document={c.doc_id} page={c.page} type={c.kind}"
        if c.image_path:
            parts.append(text_part(header + "\n(page image below)"))
            parts.append(image_part(c.image_path))
        else:
            parts.append(text_part(f"{header}\n{c.text}"))
    parts.append(text_part(f"Question: {question}"))
    return parts


def parse_citations(text: str, contexts: list[Context]) -> tuple[list[Citation], list[int]]:
    by_sid = {c.sid: c for c in contexts}
    valid, invalid, seen = [], [], set()
    for m in CITATION_RE.finditer(text):
        sid = int(m.group(1))
        if sid in seen:
            continue
        seen.add(sid)
        if sid in by_sid:
            valid.append(Citation(sid, by_sid[sid].doc_id, by_sid[sid].page))
        else:
            invalid.append(sid)
    return valid, invalid


def answer(question: str, retrieved: Retrieved, vlm: VLM) -> Answer:
    t0 = time.perf_counter()
    if not retrieved.contexts:
        text = NOT_FOUND
    else:
        text = vlm.chat(SYSTEM, build_parts(question, retrieved.contexts))
    citations, invalid = parse_citations(text, retrieved.contexts)
    return Answer(
        approach=retrieved.approach,
        question=question,
        text=text,
        citations=citations,
        contexts=retrieved.contexts,
        ranked_pages=retrieved.ranked_pages,
        not_found=NOT_FOUND in text,
        invalid_citations=invalid,
        latency_s=round(time.perf_counter() - t0, 3),
    )
