"""Extraction : texte, tableaux et figures avec Docling, plus rendu des pages en images.

Chaque élément produit reste rattaché à une seule page et garde sa zone (bbox),
pour que chaque citation renvoie à un endroit vérifiable du PDF.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pypdfium2 as pdfium

from ..models import Element

if TYPE_CHECKING:  # docling n'est requis que pour l'extraction réelle
    from docling_core.types.doc import DoclingDocument


# --------------------------------------------------------------------------- rendu des pages
def render_pages(pdf_path: Path, out_dir: Path, scale: float = 1.5) -> list[Path]:
    """Rend chaque page en PNG : data/pages/<doc_id>/<page>.png (page à partir de 1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    paths = []
    try:
        for i in range(len(pdf)):
            path = out_dir / f"{i + 1}.png"
            if not path.exists():
                pdf[i].render(scale=scale).to_pil().save(path)
            paths.append(path)
    finally:
        pdf.close()
    return paths


# --------------------------------------------------------------------------- découpage du texte
def chunk_text_blocks(
    blocks: Iterable[tuple[int, str | None, str]], doc_id: str, max_chars: int, source: str
) -> list[Element]:
    """Regroupe des blocs (page, titre, texte) consécutifs en morceaux.

    Un morceau ne déborde jamais sur une autre page ni sur une autre section :
    la citation (doc, page) reste exacte.
    """
    chunks: list[Element] = []
    cur: list[str] = []
    cur_page: int | None = None
    cur_heading: str | None = None

    def flush():
        if cur and cur_page is not None:
            body = "\n".join(cur).strip()
            if body:
                prefix = f"{cur_heading}\n" if cur_heading else ""
                chunks.append(
                    Element(doc_id, cur_page, "text", prefix + body, source=source, heading=cur_heading)
                )
        cur.clear()

    for page, heading, text in blocks:
        text = text.strip()
        if not text:
            continue
        size = sum(len(t) for t in cur)
        if page != cur_page or heading != cur_heading or size + len(text) > max_chars:
            flush()
            cur_page, cur_heading = page, heading
        # un bloc trop long est découpé seul
        while len(text) > max_chars:
            cut = text.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            cur.append(text[:cut])
            flush()
            text = text[cut:].strip()
        cur.append(text)
    flush()
    return chunks


# --------------------------------------------------------------------------- Docling
def convert_with_docling(pdf_path: Path, ocr: bool = False) -> "DoclingDocument":
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = ocr
    opts.do_table_structure = True
    opts.generate_page_images = True  # nécessaire pour découper les tableaux
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return converter.convert(str(pdf_path)).document


def _bbox(item, doc) -> tuple[int, list[float] | None]:
    prov = item.prov[0]
    page = prov.page_no
    try:
        height = doc.pages[page].size.height
        b = prov.bbox.to_top_left_origin(page_height=height)
        return page, [round(b.l, 1), round(b.t, 1), round(b.r, 1), round(b.b, 1)]
    except Exception:
        return page, None


def elements_from_docling(doc: "DoclingDocument", doc_id: str, crops_dir: Path, max_chars: int) -> list[Element]:
    """Transforme un DoclingDocument en éléments texte / tableau / figure."""
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem
    from docling_core.types.doc import DocItemLabel

    skip_labels = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
    crops_dir.mkdir(parents=True, exist_ok=True)
    blocks: list[tuple[int, str | None, str]] = []
    visuals: list[Element] = []
    heading: str | None = None
    caption_refs: set[str] = set()

    # les légendes sont rattachées aux tableaux/figures : on ne les compte pas deux fois
    for item, _ in doc.iterate_items():
        if isinstance(item, (TableItem, PictureItem)):
            caption_refs.update(c.cref for c in item.captions)

    n_tab = n_fig = 0
    for item, _ in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        if isinstance(item, SectionHeaderItem):
            heading = item.text.strip()
            continue
        if isinstance(item, TableItem):
            n_tab += 1
            page, bbox = _bbox(item, doc)
            caption = item.caption_text(doc).strip()
            md = item.export_to_markdown(doc=doc)
            img_path = _save_crop(item, doc, crops_dir / f"table_{n_tab}_p{page}.png")
            content = (caption + "\n\n" if caption else "") + md
            visuals.append(Element(doc_id, page, "table", content, "docling", bbox, img_path, heading))
            continue
        if isinstance(item, PictureItem):
            n_fig += 1
            page, bbox = _bbox(item, doc)
            caption = item.caption_text(doc).strip()
            img_path = _save_crop(item, doc, crops_dir / f"figure_{n_fig}_p{page}.png")
            content = caption or f"Figure on page {page}"
            visuals.append(Element(doc_id, page, "figure", content, "docling", bbox, img_path, heading))
            continue
        if isinstance(item, TextItem):
            if item.label in skip_labels or item.self_ref in caption_refs:
                continue
            page, _ = _bbox(item, doc)
            blocks.append((page, heading, item.text))

    return chunk_text_blocks(blocks, doc_id, max_chars, "docling") + visuals


def _save_crop(item, doc, path: Path) -> str | None:
    try:
        img = item.get_image(doc)
    except Exception:
        img = None
    if img is None:
        return None
    img.save(path)
    return str(path)


# --------------------------------------------------------------------------- extraction simple (repli)
def elements_simple(pdf_path: Path, doc_id: str, max_chars: int) -> list[Element]:
    """Texte brut page par page avec pypdfium2 : sans tableaux ni figures.

    Sert de référence « texte naïf » et de repli quand Docling n'est pas installé.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    blocks = []
    try:
        for i in range(len(pdf)):
            textpage = pdf[i].get_textpage()
            text = textpage.get_text_range()
            textpage.close()
            for para in text.replace("\r", "").split("\n\n"):
                blocks.append((i + 1, None, " ".join(para.split())))
    finally:
        pdf.close()
    return chunk_text_blocks(blocks, doc_id, max_chars, "simple")


# --------------------------------------------------------------------------- orchestration
def parse_pdf(pdf_path: Path, doc_id: str, parsed_dir: Path, pages_dir: Path, *, parser: str = "docling",
              ocr: bool = False, max_chars: int = 1200, render_scale: float = 1.5) -> list[Element]:
    out = parsed_dir / doc_id
    out.mkdir(parents=True, exist_ok=True)
    render_pages(pdf_path, pages_dir / doc_id, scale=render_scale)
    if parser == "docling":
        doc = convert_with_docling(pdf_path, ocr=ocr)
        doc.save_as_json(out / "docling.json")
        elements = elements_from_docling(doc, doc_id, out / "crops", max_chars)
    elif parser == "simple":
        elements = elements_simple(pdf_path, doc_id, max_chars)
    else:
        raise ValueError(f"parser inconnu : {parser}")
    write_elements(out / "elements.jsonl", elements)
    return elements


def write_elements(path: Path, elements: list[Element]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in elements:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def read_elements(path: Path) -> list[Element]:
    with open(path, encoding="utf-8") as f:
        return [Element(**json.loads(line)) for line in f if line.strip()]
