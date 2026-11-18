"""Client LLM vision via une API compatible OpenAI.

Le même code fonctionne avec Ollama en local (http://localhost:11434/v1),
vLLM, OpenRouter ou OpenAI : seuls l'URL, la clé et le nom du modèle changent.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Protocol

from PIL import Image


class VLM(Protocol):
    def chat(self, system: str, parts: list[dict]) -> str: ...


def image_part(path: str | Path, max_side: int = 1280) -> dict:
    """Encode une image en data URL, réduite pour limiter les tokens."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


class OpenAICompatibleVLM:
    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 512, timeout: float = 300):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, system: str, parts: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": parts}],
        )
        return (resp.choices[0].message.content or "").strip()


class EchoVLM:
    """VLM factice pour les tests : cite la première source et recopie son début."""

    def __init__(self):
        self.calls: list[tuple[str, list[dict]]] = []

    def chat(self, system: str, parts: list[dict]) -> str:
        self.calls.append((system, parts))
        texts = [p["text"] for p in parts if p["type"] == "text"]
        if "Describe" in system:
            return "Synthetic description. " + (texts[0][:200] if texts else "")
        sources = [t for t in texts if t.startswith("[S")]
        if not sources:
            return "NOT_FOUND"
        first = sources[0]
        label = first.split("]")[0] + "]"
        body = first.split("\n", 1)[1] if "\n" in first else ""
        return f"{body[:300]} {label}".strip()


def make_vlm(settings) -> VLM:
    if settings.vlm_model == "echo":
        return EchoVLM()
    return OpenAICompatibleVLM(settings.vlm_base_url, settings.vlm_api_key, settings.vlm_model,
                               settings.vlm_max_tokens, settings.vlm_timeout_s)
