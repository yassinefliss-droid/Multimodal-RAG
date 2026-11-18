"""Configuration centralisée, lue depuis les variables d'environnement ou un fichier .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MMRAG_", extra="ignore")

    # Base de données
    database_url: str = "postgresql://mmrag:mmrag@localhost:5432/mmrag"

    # Dossiers de données
    data_dir: Path = Path("data")

    # Jeu d'évaluation (lu aussi par la web app)
    questions_path: Path = Path("evals/questions.jsonl")

    # Embeddings texte (sentence-transformers) ; "hash" = embedder hors-ligne pour les tests
    text_embedder: str = "BAAI/bge-small-en-v1.5"
    text_dim: int = 384

    # Embeddings de pages (ColQwen2 via colpali-engine) ; "fake" pour les tests
    page_embedder: str = "vidore/colqwen2-v1.0"
    page_dim: int = 128
    device: str = "cpu"  # "cuda" sur Colab / GPU

    # LLM vision, via n'importe quelle API compatible OpenAI (Ollama, vLLM, OpenRouter, OpenAI...)
    vlm_base_url: str = "http://localhost:11434/v1"
    vlm_api_key: str = "ollama"
    vlm_model: str = "qwen2.5vl:7b"
    vlm_max_tokens: int = 512
    vlm_timeout_s: float = 300.0

    # Extraction
    docling_ocr: bool = False  # les PDF arXiv ont une couche texte
    page_render_scale: float = 1.5
    chunk_max_chars: int = 1200

    # Recherche
    k_retrieve: int = 20   # profondeur pour mesurer la recherche
    k_context_chunks: int = 6
    k_context_pages: int = 3
    page_prefilter: int = 40  # pages candidates avant le MaxSim

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"


@lru_cache
def get_settings() -> Settings:
    return Settings()
