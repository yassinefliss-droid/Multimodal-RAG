"""Évaluation : recherche, preuve dans le contexte, et exactitude de la réponse.

Pour chaque question et chaque approche on mesure trois choses distinctes
(une bonne page retrouvée ne garantit pas une bonne réponse) :
1. hit@k                : une page de référence est dans les k premières pages classées ;
2. evidence_in_context  : une page de référence fait partie des preuves envoyées au LLM ;
3. correct              : la réponse contient la valeur attendue (vérification automatique),
                          complétée par les verdicts humains saisis dans la web app.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import Answer

KS = (1, 3, 5, 10)
NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")  # pas de signe : « 10-20 » ne doit pas donner -20


@dataclass
class Question:
    id: str
    question: str
    doc_id: str
    pages: list[int]
    modality: str  # text | table | figure
    answer: str
    match: dict  # {"type": "number", "values": [28.4]} ou {"type": "keywords", "values": ["..."]}
    verified: bool = False

    @property
    def gold(self) -> set[tuple[str, int]]:
        return {(self.doc_id, p) for p in self.pages}


def load_questions(path: Path, allow_unverified: bool = False) -> list[Question]:
    qs = [Question(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    no_pages = [q.id for q in qs if q.verified and not q.pages]
    if no_pages:
        raise ValueError(f"questions vérifiées sans page de référence : {', '.join(no_pages)}")
    unverified = [q.id for q in qs if not q.verified]
    if unverified and not allow_unverified:
        raise ValueError(
            f"{len(unverified)} question(s) non vérifiées ({', '.join(unverified[:5])}...). "
            "Vérifie la réponse et la page dans le PDF puis mets \"verified\": true, "
            "ou relance avec --allow-unverified pour un essai."
        )
    return qs


def numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in NUM_RE.findall(text)]


def is_correct(answer_text: str, match: dict) -> bool:
    if "NOT_FOUND" in answer_text:
        return False
    if match["type"] == "number":
        found = numbers_in(answer_text)
        return all(any(abs(f - v) <= max(1e-6, 0.005 * abs(v)) for f in found) for v in match["values"])
    if match["type"] == "keywords":
        low = answer_text.lower()
        return all(str(v).lower() in low for v in match["values"])
    raise ValueError(f"type de vérification inconnu : {match['type']}")


def score(q: Question, a: Answer) -> dict:
    ranked = a.ranked_pages
    return {
        "question_id": q.id,
        "modality": q.modality,
        "approach": a.approach,
        **{f"hit@{k}": any(p in q.gold for p in ranked[:k]) for k in KS},
        "evidence_in_context": any((c.doc_id, c.page) in q.gold for c in a.contexts),
        "correct": is_correct(a.text, q.match),
        "citation_ok": any((c.doc_id, c.page) in q.gold for c in a.citations),
        "not_found": a.not_found,
        "invalid_citations": len(a.invalid_citations),
        "latency_s": a.latency_s,
        "answer": a.text,
        "cited_pages": [f"{c.doc_id}:p{c.page}" for c in a.citations],
    }


def aggregate(rows: list[dict]) -> dict:
    def summary(rs: list[dict]) -> dict:
        if not rs:
            return {}
        out = {"n": len(rs)}
        for key in [f"hit@{k}" for k in KS] + ["evidence_in_context", "correct", "citation_ok", "not_found"]:
            if key in rs[0]:
                out[key] = round(sum(r[key] for r in rs) / len(rs), 3)
        out["latency_median_s"] = round(statistics.median(r["latency_s"] for r in rs), 2)
        return out

    result: dict = {}
    for approach in sorted({r["approach"] for r in rows}):
        rs = [r for r in rows if r["approach"] == approach]
        result[approach] = {"all": summary(rs)}
        for mod in sorted({r["modality"] for r in rs}):
            result[approach][mod] = summary([r for r in rs if r["modality"] == mod])
    return result


def run_eval(pipeline, questions: list[Question], approaches: list[str], out_dir: Path,
             questions_path: Path | None = None) -> dict:
    rows = []
    for q in questions:
        for approach in approaches:
            a = pipeline.ask(q.question, approach, gold_pages=sorted(q.gold))
            r = score(q, a)
            rows.append(r)
            print(f"{q.id:6s} {approach:9s} hit@5={int(r['hit@5'])} ctx={int(r['evidence_in_context'])} "
                  f"ok={int(r['correct'])}  {a.text[:70]!r}")
    s = pipeline.settings
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest() if questions_path else None,
        "n_questions": len(questions),
        "config": {"text_embedder": s.text_embedder, "page_embedder": s.page_embedder, "vlm_model": s.vlm_model,
                   "k_context_chunks": s.k_context_chunks, "k_context_pages": s.k_context_pages,
                   "page_prefilter": s.page_prefilter},
        "summary": aggregate(rows),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"eval-{stamp}.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    return report


def to_markdown(report: dict) -> str:
    lines = [f"# Évaluation du {report['created_at']}", "",
             f"{report['n_questions']} questions · LLM vision `{report['config']['vlm_model']}` · "
             f"texte `{report['config']['text_embedder']}` · pages `{report['config']['page_embedder']}`", "",
             "| approche | sous-ensemble | n | hit@1 | hit@5 | preuve dans le contexte | réponse exacte | citation correcte | latence médiane (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for approach, parts in report["summary"].items():
        for name, m in parts.items():
            if not m:
                continue
            lines.append(f"| {approach} | {name} | {m['n']} | {m.get('hit@1', '')} | {m.get('hit@5', '')} | "
                         f"{m['evidence_in_context']} | {m['correct']} | {m['citation_ok']} | {m['latency_median_s']} |")
    lines += ["", "Les valeurs sont des proportions (0 à 1). « réponse exacte » est la vérification automatique ; "
              "les verdicts humains de la web app sont exportés à part (`mmrag judgments`)."]
    return "\n".join(lines) + "\n"


def asdict_question(q: Question) -> dict:
    return asdict(q)
