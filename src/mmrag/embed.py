"""Embeddings texte (un vecteur par morceau) et de pages (plusieurs vecteurs par page).

- Texte : sentence-transformers, BGE-small par défaut.
- Pages : ColQwen2 (famille ColPali), qui produit un vecteur par patch d'image ;
  la similarité requête-page est le MaxSim (interaction tardive).
Des versions factices, déterministes et hors-ligne, servent aux tests.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class TextEmbedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class PageEmbedder(Protocol):
    dim: int

    def embed_pages(self, image_paths: list[Path]) -> list[np.ndarray]: ...  # chacun (n_tokens, dim)
    def embed_query(self, text: str) -> np.ndarray: ...  # (n_tokens, dim)


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


# --------------------------------------------------------------------------- texte
class SentenceTransformerEmbedder:
    def __init__(self, name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in name.lower() and "m3" not in name.lower() else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def _hash_vec(token: str, dim: int) -> np.ndarray:
    h = hashlib.sha256(token.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.standard_normal(dim)


class HashEmbedder:
    """Sac de mots haché : aucune dépendance ni téléchargement. Pour les tests uniquement."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for t in _tokens(text):
            v += _hash_vec(t, self.dim)
        return _normalize(v) if v.any() else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


# --------------------------------------------------------------------------- pages
class ColQwenEmbedder:
    def __init__(self, name: str, device: str = "cuda"):
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self.torch = torch
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = ColQwen2.from_pretrained(name, torch_dtype=dtype, device_map=device).eval()
        self.processor = ColQwen2Processor.from_pretrained(name)
        self.device = device
        self.dim = 128

    def _run(self, batch) -> list[np.ndarray]:
        batch = batch.to(self.device)
        with self.torch.no_grad():
            out = self.model(**batch)
        mask = batch["attention_mask"].bool()
        return [out[i][mask[i]].float().cpu().numpy() for i in range(out.shape[0])]

    def embed_pages(self, image_paths: list[Path], batch_size: int = 4) -> list[np.ndarray]:
        from PIL import Image

        result = []
        for i in range(0, len(image_paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]
            result += self._run(self.processor.process_images(images))
        return result

    def embed_query(self, text: str) -> np.ndarray:
        return self._run(self.processor.process_queries([text]))[0]


class FakePageEmbedder:
    """Remplace ColQwen2 dans les tests : « lit » le texte de la page via un fichier .txt voisin.

    Chaque mot devient un vecteur de token, ce qui permet de tester tout le
    chemin MaxSim dans pgvector sans modèle ni GPU.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _multi(self, text: str) -> np.ndarray:
        toks = _tokens(text)[:256] or ["empty"]
        return _normalize(np.stack([_hash_vec(t, self.dim) for t in toks]))

    def embed_pages(self, image_paths: list[Path]) -> list[np.ndarray]:
        out = []
        for p in image_paths:
            sidecar = Path(p).with_suffix(".txt")
            out.append(self._multi(sidecar.read_text(encoding="utf-8") if sidecar.exists() else ""))
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self._multi(text)


def make_text_embedder(settings) -> TextEmbedder:
    if settings.text_embedder == "hash":
        return HashEmbedder(settings.text_dim)
    return SentenceTransformerEmbedder(settings.text_embedder, settings.device)


def make_page_embedder(settings) -> PageEmbedder:
    if settings.page_embedder == "fake":
        return FakePageEmbedder(settings.page_dim)
    return ColQwenEmbedder(settings.page_embedder, settings.device)
