"""Les approches de recherche comparées.

| approche  | ce qui est cherché                                   | ce que lit le LLM vision |
|-----------|------------------------------------------------------|--------------------------|
| text      | texte Docling + tableaux en Markdown (hybride + RRF) | du texte                 |
| captions  | idem + descriptions VLM des figures et tableaux      | du texte                 |
| pages     | pages en images (ColQwen2, MaxSim dans pgvector)     | les images des pages     |
| hybrid    | captions + pages, fusionnées par RRF                 | texte + images           |
| oracle    | (évaluation) les pages de référence, sans recherche  | les images des pages     |

L'approche « oracle » sert de contrôle : si elle échoue, le problème vient de la
lecture du LLM vision, pas de la recherche.
"""
from __future__ import annotations

from collections import OrderedDict

from .models import Context, Retrieved

APPROACHES = ["text", "captions", "pages", "hybrid"]
TEXT_SOURCES = {"text": ["docling", "simple"], "captions": ["docling", "simple", "vlm_description"]}


def rrf(rankings: list[list], k: int = 60) -> list[tuple[object, float]]:
    """Reciprocal Rank Fusion : score(d) = somme des 1 / (k + rang)."""
    scores: dict = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _dedupe_pages(pairs) -> list[tuple[str, int]]:
    return list(OrderedDict.fromkeys(pairs))


class Retriever:
    def __init__(self, store, text_embedder, page_embedder, settings):
        self.store = store
        self.text_embedder = text_embedder
        self.page_embedder = page_embedder
        self.s = settings

    # ------------------------------------------------------------------ briques
    def _chunks(self, question: str, sources: list[str], k: int) -> list[dict]:
        qvec = self.text_embedder.embed_query(question)
        dense = self.store.search_vector(qvec, sources, k)
        sparse = self.store.search_fulltext(question, sources, k)
        by_id = {r["id"]: r for r in dense + sparse}
        fused = rrf([[r["id"] for r in dense], [r["id"] for r in sparse]])
        return [{**by_id[i], "score": s} for i, s in fused[:k]]

    def _pages(self, question: str, k: int) -> list[dict]:
        if self.page_embedder is None:
            return []
        qtok = self.page_embedder.embed_query(question)
        return self.store.search_pages_maxsim(qtok, k, self.s.page_prefilter)

    @staticmethod
    def _chunk_contexts(rows: list[dict], start: int = 1) -> list[Context]:
        return [Context(start + i, r["doc_id"], r["page"], r["kind"], text=r["content"], score=r["score"])
                for i, r in enumerate(rows)]

    @staticmethod
    def _page_contexts(rows: list[dict], start: int = 1) -> list[Context]:
        return [Context(start + i, r["doc_id"], r["page"], "page", image_path=r["image_path"], score=r["score"])
                for i, r in enumerate(rows)]

    # ------------------------------------------------------------------ approches
    def retrieve(self, question: str, approach: str, gold_pages: list[tuple[str, int]] | None = None) -> Retrieved:
        K = self.s.k_retrieve
        if approach in ("text", "captions"):
            rows = self._chunks(question, TEXT_SOURCES[approach], K)
            ctx = self._chunk_contexts(rows[: self.s.k_context_chunks])
            return Retrieved(approach, ctx, _dedupe_pages((r["doc_id"], r["page"]) for r in rows))

        if approach == "pages":
            rows = self._pages(question, K)
            ctx = self._page_contexts(rows[: self.s.k_context_pages])
            return Retrieved(approach, ctx, [(r["doc_id"], r["page"]) for r in rows])

        if approach == "hybrid":
            chunk_rows = self._chunks(question, TEXT_SOURCES["captions"], K)
            page_rows = self._pages(question, K)
            fused = rrf([
                _dedupe_pages((r["doc_id"], r["page"]) for r in chunk_rows),
                [(r["doc_id"], r["page"]) for r in page_rows],
            ])
            ranked = [key for key, _ in fused]
            # contexte : les meilleurs morceaux de texte + les meilleures pages en images
            n_img = max(1, self.s.k_context_pages - 1)
            text_ctx = self._chunk_contexts(chunk_rows[: self.s.k_context_chunks - 2])
            top_pages = [p for p in ranked if self.store.page_image(*p)][:n_img]
            imgs = [{"doc_id": d, "page": p, "image_path": self.store.page_image(d, p), "score": 0.0}
                    for d, p in top_pages]
            return Retrieved(approach, text_ctx + self._page_contexts(imgs, start=len(text_ctx) + 1), ranked)

        if approach == "oracle":
            if not gold_pages:
                raise ValueError("l'approche oracle demande les pages de référence")
            imgs = [{"doc_id": d, "page": p, "image_path": self.store.page_image(d, p), "score": 1.0}
                    for d, p in gold_pages]
            return Retrieved(approach, self._page_contexts([i for i in imgs if i["image_path"]]), list(gold_pages))

        raise ValueError(f"approche inconnue : {approach}")
