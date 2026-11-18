"""Approche « descriptions » : le LLM vision décrit chaque figure et chaque tableau.

Les descriptions sont indexées comme du texte, ce qui rend cherchables des
informations qui n'existent que dans l'image (valeurs d'un graphique, schéma...).
"""
from __future__ import annotations

from ..models import Element
from ..vlm import VLM, image_part, text_part

SYSTEM = (
    "Describe this element from a scientific paper so it can be found by search. "
    "State what it is (chart type, table, diagram), its title, axes, legend and units, "
    "and transcribe every number, label and model name you can read. "
    "Do not invent values you cannot read. Answer in English, at most 200 words."
)


def describe_elements(elements: list[Element], vlm: VLM) -> list[Element]:
    out = []
    for e in elements:
        if e.kind not in ("figure", "table") or not e.image_path:
            continue
        hint = f"Caption or extracted content:\n{e.content[:1500]}"
        desc = vlm.chat(SYSTEM, [image_part(e.image_path, max_side=1024), text_part(hint)])
        caption = e.content.split("\n", 1)[0] if e.kind == "table" else e.content
        out.append(Element(e.doc_id, e.page, e.kind, f"{caption}\n{desc}".strip(), "vlm_description",
                           e.bbox, e.image_path, e.heading))
    return out
