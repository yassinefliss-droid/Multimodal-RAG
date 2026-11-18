"""Structures de données partagées par tout le pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Kind = Literal["text", "table", "figure"]
Source = Literal["docling", "simple", "vlm_description"]


@dataclass
class Element:
    """Un morceau de document, toujours rattaché à UNE page (citation précise)."""

    doc_id: str
    page: int  # numérotation à partir de 1, comme dans le PDF
    kind: Kind
    content: str  # texte, tableau en Markdown, ou légende/description
    source: Source = "docling"
    bbox: list[float] | None = None  # [x0, y0, x1, y1], origine en haut à gauche, en points PDF
    image_path: str | None = None  # découpe de la figure ou du tableau
    heading: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Context:
    """Une preuve envoyée au LLM vision, numérotée [S1], [S2]..."""

    sid: int
    doc_id: str
    page: int
    kind: str  # text | table | figure | page
    text: str | None = None
    image_path: str | None = None
    score: float = 0.0

    @property
    def label(self) -> str:
        return f"S{self.sid}"


@dataclass
class Retrieved:
    """Résultat d'une approche de recherche : contextes envoyés + classement des pages."""

    approach: str
    contexts: list[Context]
    ranked_pages: list[tuple[str, int]] = field(default_factory=list)  # (doc_id, page), pour hit@k


@dataclass
class Citation:
    sid: int
    doc_id: str
    page: int


@dataclass
class Answer:
    approach: str
    question: str
    text: str
    citations: list[Citation]
    contexts: list[Context]
    ranked_pages: list[tuple[str, int]]
    not_found: bool
    invalid_citations: list[int]
    latency_s: float
