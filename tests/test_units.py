from mmrag.answer import parse_citations
from mmrag.evaluate import is_correct, numbers_in
from mmrag.ingest.parse import chunk_text_blocks
from mmrag.models import Context
from mmrag.retrieve import rrf


def test_rrf_rewards_agreement():
    fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]]))
    assert fused["a"] == fused["b"] > fused["c"] > 0
    assert set(fused) == {"a", "b", "c", "d"}


def test_chunks_never_cross_pages_or_sections():
    blocks = [(1, "Intro", "alpha " * 10), (1, "Intro", "beta"), (2, "Intro", "gamma"), (2, "Method", "delta")]
    chunks = chunk_text_blocks(blocks, "doc", max_chars=1000, source="docling")
    assert [(c.page, c.heading) for c in chunks] == [(1, "Intro"), (2, "Intro"), (2, "Method")]
    assert chunks[0].content.startswith("Intro\n") and "beta" in chunks[0].content


def test_long_block_is_split():
    chunks = chunk_text_blocks([(3, None, "word " * 500)], "doc", max_chars=300, source="simple")
    assert len(chunks) > 1 and all(len(c.content) <= 300 for c in chunks) and {c.page for c in chunks} == {3}


def test_citations_are_validated_against_context():
    ctx = [Context(1, "d1", 4, "text", text="x"), Context(2, "d2", 7, "page", image_path="p.png")]
    valid, invalid = parse_citations("A [S2] and B [S1][S2] and C [S9]", ctx)
    assert [(c.doc_id, c.page) for c in valid] == [("d2", 7), ("d1", 4)]
    assert invalid == [9]


def test_numbers_and_correctness():
    assert numbers_in("reaches 28.4 BLEU with 1,024 tokens, 10-20 layers") == [28.4, 1024.0, 10.0, 20.0]
    assert is_correct("The big model gets 28.4 BLEU [S1]", {"type": "number", "values": [28.4]})
    assert not is_correct("The big model gets 27.3 BLEU [S1]", {"type": "number", "values": [28.4]})
    assert is_correct("It uses Adam with warmup", {"type": "keywords", "values": ["adam", "warmup"]})
    assert not is_correct("NOT_FOUND 28.4", {"type": "number", "values": [28.4]})
