"""Teste la conversion DoclingDocument -> éléments, sur un document construit à la main.

On n'exécute pas les modèles Docling ici (ils se téléchargent depuis Hugging Face) :
on construit directement le DoclingDocument que Docling produirait.
"""
import pytest

pytest.importorskip("docling_core")
from docling_core.types.doc import (  # noqa: E402
    BoundingBox, CoordOrigin, DocItemLabel, DoclingDocument, ImageRef, ProvenanceItem, Size, TableCell, TableData,
)
from PIL import Image  # noqa: E402

from mmrag.ingest.parse import elements_from_docling  # noqa: E402


def prov(page, l, t, r, b):
    # Docling stocke les boîtes avec l'origine en bas à gauche (repère PDF)
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.BOTTOMLEFT),
                          charspan=(0, 1))


def build_doc():
    doc = DoclingDocument(name="toy")
    img = Image.new("RGB", (60, 40), "white")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    doc.add_page(page_no=2, size=Size(width=600, height=800))
    doc.add_heading("3 Results", prov=prov(1, 50, 780, 300, 760))
    doc.add_text(DocItemLabel.PAGE_HEADER, "Preprint — do not cite", prov=prov(1, 50, 795, 300, 790))
    doc.add_text(DocItemLabel.TEXT, "Our model reaches 28.4 BLEU.", prov=prov(1, 50, 740, 550, 700))
    cap = doc.add_text(DocItemLabel.CAPTION, "Table 2: BLEU scores.", prov=prov(1, 50, 690, 550, 680))
    cells = [TableCell(text=t, start_row_offset_idx=r, end_row_offset_idx=r + 1,
                       start_col_offset_idx=c, end_col_offset_idx=c + 1, column_header=(r == 0))
             for r, row in enumerate([["Model", "BLEU"], ["Big", "28.4"]]) for c, t in enumerate(row)]
    doc.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=cells), caption=cap,
                  prov=prov(1, 50, 670, 550, 600))
    fcap = doc.add_text(DocItemLabel.CAPTION, "Figure 3: Attention heads.", prov=prov(2, 50, 300, 550, 290))
    doc.add_picture(image=ImageRef.from_pil(img, dpi=72), caption=fcap, prov=prov(2, 100, 600, 500, 320))
    return doc


def test_elements_from_docling(tmp_path):
    els = elements_from_docling(build_doc(), "toy", tmp_path / "crops", max_chars=1000)
    by_kind = {k: [e for e in els if e.kind == k] for k in ("text", "table", "figure")}

    [text] = by_kind["text"]
    assert text.page == 1 and text.heading == "3 Results"
    assert "28.4 BLEU" in text.content
    assert "Preprint" not in text.content          # en-tête de page ignoré
    assert "Table 2" not in text.content           # légende rattachée au tableau, pas dupliquée

    [table] = by_kind["table"]
    assert table.page == 1 and table.content.startswith("Table 2: BLEU scores.")
    assert "| Big" in table.content and "28.4" in table.content
    assert table.bbox == [50.0, 130.0, 550.0, 200.0]  # converti en origine haut-gauche

    [fig] = by_kind["figure"]
    assert fig.page == 2 and fig.content == "Figure 3: Attention heads."
    assert fig.image_path and (tmp_path / "crops").exists()
