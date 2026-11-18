"""Ligne de commande : mmrag <étape> [options]."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import get_settings
from .retrieve import APPROACHES


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mmrag", description="RAG multimodal sur PDF")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="télécharge les PDF listés dans data/sources.csv")
    d.add_argument("--sources", default="data/sources.csv")

    s = sub.add_parser("parse", help="extraction Docling + rendu des pages")
    s.add_argument("--parser", choices=["docling", "simple"], default="docling")
    s.add_argument("--only", nargs="*")

    sub.add_parser("describe", help="descriptions VLM des figures et tableaux").add_argument("--only", nargs="*")

    sub.add_parser("embed-pages", help="embeddings ColQwen2 des pages, mis en cache (GPU conseillé)").add_argument("--only", nargs="*")

    i = sub.add_parser("index", help="indexation dans pgvector")
    i.add_argument("--what", nargs="+", choices=["text", "pages"], default=["text", "pages"])
    i.add_argument("--only", nargs="*")

    a = sub.add_parser("ask", help="pose une question")
    a.add_argument("question")
    a.add_argument("--approach", choices=APPROACHES, default="hybrid")

    e = sub.add_parser("eval", help="évaluation sur le jeu de questions")
    e.add_argument("--questions", default=None, help="par défaut : MMRAG_QUESTIONS_PATH")
    e.add_argument("--approaches", nargs="+", default=APPROACHES + ["oracle"])
    e.add_argument("--allow-unverified", action="store_true")
    e.add_argument("--out", default="evals/results")

    sub.add_parser("judgments", help="exporte les verdicts humains de la web app")
    sub.add_parser("stats", help="contenu de l'index")

    v = sub.add_parser("serve", help="lance l'API et la web app")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    settings = get_settings()

    if args.cmd == "download":
        from .ingest.download import download_all

        download_all(Path(args.sources), settings.pdf_dir)
    elif args.cmd == "parse":
        from .pipeline import run_parse

        run_parse(settings, args.parser, args.only)
    elif args.cmd == "describe":
        from .pipeline import run_describe

        run_describe(settings, args.only)
    elif args.cmd == "embed-pages":
        from .pipeline import run_embed_pages

        run_embed_pages(settings, args.only)
    elif args.cmd == "index":
        from .pipeline import run_index

        run_index(settings, tuple(args.what), args.only)
    elif args.cmd == "ask":
        from .pipeline import Pipeline

        ans = Pipeline(settings, with_pages=args.approach in ("pages", "hybrid")).ask(args.question, args.approach)
        print(ans.text)
        for c in ans.citations:
            print(f"  [S{c.sid}] {c.doc_id} page {c.page}")
        print(f"({ans.latency_s:.1f} s)")
    elif args.cmd == "eval":
        from .evaluate import load_questions, run_eval
        from .pipeline import Pipeline

        qpath = Path(args.questions) if args.questions else settings.questions_path
        qs = load_questions(qpath, args.allow_unverified)
        run_eval(Pipeline(settings), qs, args.approaches, Path(args.out), qpath)
    elif args.cmd == "judgments":
        from .store import default_store

        print(json.dumps(default_store(settings).judgments(), indent=2, default=str, ensure_ascii=False))
    elif args.cmd == "stats":
        from .store import default_store

        print(json.dumps(default_store(settings).stats(), indent=2))
    elif args.cmd == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
