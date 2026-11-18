"""Test de bout en bout sur des PDF synthétiques, avec Postgres + pgvector réels.

Les modèles sont remplacés par des versions factices (embedder haché, faux
ColQwen qui lit un .txt à côté de l'image, VLM « écho ») : on teste le câblage
(extraction, index, MaxSim SQL, RRF, citations, évaluation, API), pas la qualité.

Base requise : MMRAG_TEST_DATABASE_URL (par défaut postgres local sur le port 5433).
"""
import json
import os

import psycopg
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from mmrag.config import Settings
from mmrag.evaluate import Question, run_eval
from mmrag.ingest.parse import read_elements, write_elements
from mmrag.models import Element
from mmrag.pipeline import Pipeline, run_describe, run_index, run_parse

DB = os.environ.get("MMRAG_TEST_DATABASE_URL", "postgresql://postgres@127.0.0.1:5433/mmrag_test")


def _db_available() -> bool:
    try:
        with psycopg.connect(DB, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="Postgres de test indisponible")


def make_pdfs(pdf_dir):
    pdf_dir.mkdir(parents=True)
    c = canvas.Canvas(str(pdf_dir / "alpha.pdf"), pagesize=A4)
    c.drawString(72, 760, "1 Introduction")
    c.drawString(72, 740, "The Transformer relies entirely on attention mechanisms.")
    c.showPage()
    c.drawString(72, 760, "Table 1: Translation results on WMT 2014 English-German")
    for i, row in enumerate([["Model", "BLEU"], ["Transformer big", "28.4"], ["ByteNet", "23.75"]]):
        c.drawString(72, 730 - 18 * i, row[0]); c.drawString(260, 730 - 18 * i, row[1])
    c.showPage(); c.save()

    c = canvas.Canvas(str(pdf_dir / "beta.pdf"), pagesize=A4)
    c.drawString(72, 760, "Figure 1: Top-1 accuracy by model")
    for i, h in enumerate([120, 180, 150]):  # barres : les valeurs ne sont PAS dans la couche texte
        c.rect(100 + 80 * i, 450, 50, h, fill=1)
    c.showPage(); c.save()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("mmrag")
    with psycopg.connect(DB, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS judgments, page_tokens, pages, chunks, documents CASCADE")
    s = Settings(database_url=DB, data_dir=root / "data", text_embedder="hash", text_dim=64,
                 page_embedder="fake", page_dim=32, vlm_model="echo", page_prefilter=0,
                 questions_path=root / "questions.jsonl", _env_file=None)
    make_pdfs(s.pdf_dir)
    run_parse(s, parser="simple")

    # ce que « verrait » le modèle de pages : un .txt à côté de chaque image (lu par FakePageEmbedder)
    (s.pages_dir / "alpha" / "1.txt").write_text("introduction transformer attention")
    (s.pages_dir / "alpha" / "2.txt").write_text("table translation BLEU transformer big 28.4 bytenet")
    (s.pages_dir / "beta" / "1.txt").write_text("bar chart top-1 accuracy resnet 91.7 vit 88.1")

    # une figure, comme Docling l'aurait extraite, pour tester les descriptions VLM
    els = read_elements(s.parsed_dir / "beta" / "elements.jsonl")
    els.append(Element("beta", 1, "figure", "Figure 1: Top-1 accuracy by model", "simple",
                       image_path=str(s.pages_dir / "beta" / "1.png")))
    write_elements(s.parsed_dir / "beta" / "elements.jsonl", els)
    run_describe(s)
    run_index(s)
    return s


def test_index_contents(env):
    stats = Pipeline(env).store.stats()
    assert stats["documents"] == 2 and stats["pages"] == 3
    assert stats["chunks"]["vlm_description/figure"] == 1
    assert stats["page_tokens"] > 0


def test_each_approach_retrieves_and_cites(env):
    p = Pipeline(env)
    q = "What BLEU score does the Transformer big model reach on WMT 2014 English-German?"
    for approach in ("text", "captions", "pages", "hybrid"):
        a = p.ask(q, approach)
        assert ("alpha", 2) in a.ranked_pages[:2], (approach, a.ranked_pages)
        assert a.contexts and a.citations, approach
        assert not a.invalid_citations


def test_page_maxsim_finds_visual_only_content(env):
    a = Pipeline(env).ask("Which model has the best top-1 accuracy in the bar chart?", "pages")
    assert a.ranked_pages[0] == ("beta", 1)


def test_maxsim_prefilter_matches_exact(env):
    p = Pipeline(env)
    qt = p.retriever.page_embedder.embed_query("translation BLEU table")
    exact = p.store.search_pages_maxsim(qt, k=3, prefilter=0)
    pre = p.store.search_pages_maxsim(qt, k=3, prefilter=3)
    assert [(r["doc_id"], r["page"]) for r in exact] == [(r["doc_id"], r["page"]) for r in pre]


def test_eval_report(env, tmp_path):
    qs = [Question("q1", "What BLEU does Transformer big reach on WMT 2014 English-German?", "alpha", [2],
                   "table", "28.4", {"type": "number", "values": [28.4]}, True),
          Question("q2", "Top-1 accuracy of ResNet in Figure 1?", "beta", [1], "figure", "91.7",
                   {"type": "number", "values": [91.7]}, True)]
    report = run_eval(Pipeline(env), qs, ["text", "pages", "oracle"], tmp_path)
    s = report["summary"]
    assert s["text"]["table"]["hit@1"] == 1.0 and s["text"]["table"]["correct"] == 1.0  # « 28.4 » est dans le texte
    assert s["text"]["figure"]["correct"] == 0.0     # la valeur n'existe que dans l'image
    assert s["oracle"]["all"]["evidence_in_context"] == 1.0
    assert list(tmp_path.glob("eval-*.md"))


def test_api(env):
    from fastapi.testclient import TestClient

    from mmrag.api import create_app

    env.questions_path.write_text(json.dumps({"id": "q1", "question": "BLEU?", "doc_id": "alpha", "pages": [2],
                                              "modality": "table", "answer": "28.4",
                                              "match": {"type": "number", "values": [28.4]}}) + "\n")
    client = TestClient(create_app(Pipeline(env)))
    assert "RAG multimodal" in client.get("/").text
    r = client.post("/compare", json={"question": "Transformer big BLEU score", "approaches": ["text", "pages"]})
    assert r.status_code == 200
    answers = r.json()["answers"]
    assert {a["approach"] for a in answers} == {"text", "pages"}
    img = answers[0]["contexts"][0]["image_url"]
    assert client.get(img).headers["content-type"] == "image/png"
    assert client.post("/compare", json={"question": "abc", "approaches": ["nope"]}).status_code == 422
    assert client.get("/pages/..%2F..%2Fetc/1").status_code == 404
    j = client.post("/judgments", json={"question": "q", "approach": "text", "answer": "a", "verdict": "correct"})
    assert j.status_code == 200
    assert client.get("/questions").json()[0]["id"] == "q1"
