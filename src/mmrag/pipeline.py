"""Orchestration : extraction -> descriptions -> indexation, et question -> réponse."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .answer import answer
from .embed import make_page_embedder, make_text_embedder
from .ingest.describe import describe_elements
from .ingest.parse import parse_pdf, read_elements, write_elements
from .models import Answer, Element
from .retrieve import Retriever
from .store import default_store
from .vlm import make_vlm


def _manifest(settings) -> dict:
    path = settings.data_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _doc_ids(settings, only: list[str] | None) -> list[str]:
    ids = sorted(p.stem for p in settings.pdf_dir.glob("*.pdf"))
    return [i for i in ids if not only or i in only]


def run_parse(settings, parser: str = "docling", only: list[str] | None = None) -> None:
    for doc_id in _doc_ids(settings, only):
        t0 = time.perf_counter()
        els = parse_pdf(settings.pdf_dir / f"{doc_id}.pdf", doc_id, settings.parsed_dir, settings.pages_dir,
                        parser=parser, ocr=settings.docling_ocr, max_chars=settings.chunk_max_chars,
                        render_scale=settings.page_render_scale)
        kinds = {k: sum(e.kind == k for e in els) for k in ("text", "table", "figure")}
        print(f"{doc_id:12s} {kinds}  ({time.perf_counter() - t0:.0f} s)")


def run_describe(settings, only: list[str] | None = None) -> None:
    vlm = make_vlm(settings)
    for doc_id in _doc_ids(settings, only):
        src = settings.parsed_dir / doc_id / "elements.jsonl"
        dst = settings.parsed_dir / doc_id / "descriptions.jsonl"
        if dst.exists():
            print(f"{doc_id:12s} descriptions déjà faites")
            continue
        descs = describe_elements(read_elements(src), vlm)
        write_elements(dst, descs)
        print(f"{doc_id:12s} {len(descs)} descriptions")


def run_index(settings, what: tuple[str, ...] = ("text", "pages"), only: list[str] | None = None) -> None:
    store = default_store(settings)
    store.init()
    manifest = _manifest(settings)
    text_emb = make_text_embedder(settings) if "text" in what else None
    if text_emb is not None and text_emb.dim != settings.text_dim:
        raise ValueError(f"MMRAG_TEXT_DIM={settings.text_dim} mais le modèle produit {text_emb.dim} dimensions")
    page_emb = None  # chargé seulement si un embedding de page manque en cache
    for doc_id in _doc_ids(settings, only):
        page_paths = sorted((settings.pages_dir / doc_id).glob("*.png"), key=lambda p: int(p.stem))
        meta = manifest.get(doc_id, {})
        store.upsert_document(doc_id, meta.get("title"), meta.get("sha256"), len(page_paths))
        if text_emb is not None:
            for fname, sources in (("elements.jsonl", ["docling", "simple"]), ("descriptions.jsonl", ["vlm_description"])):
                path = settings.parsed_dir / doc_id / fname
                if not path.exists():
                    continue
                els: list[Element] = read_elements(path)
                vecs = text_emb.embed_documents([e.content for e in els])
                store.replace_chunks(doc_id, sources, els, vecs)
        if "pages" in what:
            for path, tokens in zip(page_paths, _page_tokens(settings, doc_id, page_paths, page_emb)):
                store.replace_page(doc_id, int(path.stem), str(path), tokens)
        print(f"{doc_id:12s} indexé")
    print(store.stats())


def _cache_dir(settings, doc_id: str) -> Path:
    return settings.data_dir / "page_emb" / doc_id


def _page_tokens(settings, doc_id: str, page_paths: list[Path], page_emb) -> list:
    """Lit les embeddings de pages en cache (calculés sur GPU), sinon les calcule."""
    cache = _cache_dir(settings, doc_id)
    files = [cache / f"{p.stem}.npy" for p in page_paths]
    if page_paths and all(f.exists() for f in files):
        return [np.load(f) for f in files]
    if page_emb is None:
        page_emb = make_page_embedder(settings)
    tokens = page_emb.embed_pages(page_paths)
    cache.mkdir(parents=True, exist_ok=True)
    for f, t in zip(files, tokens):
        np.save(f, np.asarray(t, dtype=np.float16))
    return tokens


def run_embed_pages(settings, only: list[str] | None = None) -> None:
    """Calcule les embeddings ColQwen2 sans base de données (pratique sur Colab avec GPU).

    Copie ensuite data/page_emb/ sur ta machine : `mmrag index` les réutilise.
    """
    page_emb = make_page_embedder(settings)
    for doc_id in _doc_ids(settings, only):
        page_paths = sorted((settings.pages_dir / doc_id).glob("*.png"), key=lambda p: int(p.stem))
        t0 = time.perf_counter()
        _page_tokens(settings, doc_id, page_paths, page_emb)
        print(f"{doc_id:12s} {len(page_paths)} pages ({time.perf_counter() - t0:.0f} s)")


class Pipeline:
    """Objets lourds (modèles) chargés une seule fois, pour l'API et l'évaluation."""

    def __init__(self, settings, with_pages: bool = True):
        self.settings = settings
        self.store = default_store(settings)
        self.retriever = Retriever(self.store, make_text_embedder(settings),
                                   make_page_embedder(settings) if with_pages else None, settings)
        self.vlm = make_vlm(settings)

    def ask(self, question: str, approach: str, gold_pages: list[tuple[str, int]] | None = None) -> Answer:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, approach, gold_pages)
        ans = answer(question, retrieved, self.vlm)
        ans.latency_s = round(time.perf_counter() - t0, 3)
        return ans


def page_path(settings, doc_id: str, page: int) -> Path:
    return settings.pages_dir / doc_id / f"{page}.png"
