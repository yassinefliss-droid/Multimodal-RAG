"""Téléchargement du corpus arXiv listé dans data/sources.csv.

Les PDF ne sont pas versionnés (licences variables) : seule la liste l'est.
Un manifeste data/manifest.json enregistre l'empreinte SHA-256 de chaque fichier,
pour que les résultats d'évaluation soient rattachés à des PDF précis.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import requests

ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"
HEADERS = {"User-Agent": "mmrag-research/0.1 (student project)"}


def read_sources(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_all(csv_path: Path, pdf_dir: Path, delay_s: float = 3.0) -> dict:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for src in read_sources(csv_path):
        target = pdf_dir / f"{src['id']}.pdf"
        if not target.exists():
            url = ARXIV_PDF.format(arxiv_id=src["arxiv_id"])
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError(f"{url} n'a pas renvoyé un PDF")
            target.write_bytes(resp.content)
            print(f"téléchargé  {target.name}")
            time.sleep(delay_s)  # arXiv demande de limiter le rythme
        else:
            print(f"déjà là     {target.name}")
        manifest[src["id"]] = {**src, "sha256": sha256(target), "file": target.name}
    (pdf_dir.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
