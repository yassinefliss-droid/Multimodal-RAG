"""API FastAPI et web app de comparaison.

GET  /                      la page de comparaison
POST /ask                   une question, une approche
POST /compare               une question, plusieurs approches côte à côte
POST /judgments             verdict humain (correct / partial / wrong)
GET  /questions             le jeu d'évaluation, pour le choisir dans la page
GET  /pages/{doc}/{page}    l'image d'une page citée
GET  /health                état de l'index
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .models import Answer
from .retrieve import APPROACHES

WEB = Path(__file__).parent / "web" / "index.html"


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    approach: str = "hybrid"


class CompareIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    approaches: list[str] = Field(default_factory=lambda: list(APPROACHES))
    question_id: str | None = None


class JudgmentIn(BaseModel):
    question: str
    approach: str
    answer: str
    verdict: Literal["correct", "partial", "wrong"]
    question_id: str | None = None
    note: str | None = None


def serialize(a: Answer) -> dict:
    d = asdict(a)
    d["ranked_pages"] = [f"{doc}:p{p}" for doc, p in a.ranked_pages[:10]]
    for c in d["contexts"]:
        c["image_url"] = f"/pages/{c['doc_id']}/{c['page']}"
        c.pop("image_path", None)
        if c.get("text") and len(c["text"]) > 600:
            c["text"] = c["text"][:600] + "…"
    return d


def create_app(pipeline=None) -> FastAPI:
    settings = get_settings() if pipeline is None else pipeline.settings
    app = FastAPI(title="RAG multimodal sur PDF")
    state: dict = {"pipeline": pipeline}

    def pipe():
        if state["pipeline"] is None:  # chargement paresseux des modèles
            from .pipeline import Pipeline

            state["pipeline"] = Pipeline(settings)
        return state["pipeline"]

    def check(approach: str):
        if approach not in APPROACHES:
            raise HTTPException(422, f"approche inconnue : {approach} (choix : {APPROACHES})")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return WEB.read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "approaches": APPROACHES, "index": pipe().store.stats()}

    @app.post("/ask")
    def ask(body: AskIn):
        check(body.approach)
        return serialize(pipe().ask(body.question, body.approach))

    @app.post("/compare")
    def compare(body: CompareIn):
        for a in body.approaches:
            check(a)
        p = pipe()
        with ThreadPoolExecutor(max_workers=len(body.approaches)) as ex:
            answers = list(ex.map(lambda a: p.ask(body.question, a), body.approaches))
        return {"question": body.question, "question_id": body.question_id,
                "answers": [serialize(a) for a in answers]}

    @app.post("/judgments")
    def judge(body: JudgmentIn):
        jid = pipe().store.save_judgment(body.question_id, body.question, body.approach, body.answer,
                                         body.verdict, body.note)
        return {"id": jid}

    @app.get("/questions")
    def questions():
        path = settings.questions_path
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    @app.get("/pages/{doc_id}/{page}")
    def page_image(doc_id: str, page: int):
        path = (settings.pages_dir / doc_id / f"{page}.png").resolve()
        if settings.pages_dir.resolve() not in path.parents or not path.exists():
            raise HTTPException(404, "page introuvable")
        return FileResponse(path, media_type="image/png")

    return app


app = None  # créé par `mmrag serve` ou `uvicorn mmrag.api:create_app --factory`
