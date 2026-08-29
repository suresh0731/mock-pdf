from types import SimpleNamespace

from app.services.ocr.native_text import (
    classify_and_extract,
    classify_page,
    extract_native_words,
)


class FakePage:
    """Minimal fitz.Page double: rect + get_text("words"/"blocks")."""

    def __init__(self, width: float, height: float, words: list[tuple], blocks: list[tuple]):
        self.rect = SimpleNamespace(width=width, height=height)
        self._words = words
        self._blocks = blocks

    def get_text(self, mode: str):
        if mode == "words":
            return self._words
        if mode == "blocks":
            return self._blocks
        raise ValueError(f"unsupported mode: {mode}")


def _digital_page() -> FakePage:
    """~30 words covering a healthy fraction of a small page — clearly digital."""
    words = []
    blocks = []
    y = 10.0
    for row in range(6):
        x = 10.0
        for col in range(5):
            words.append(("word%d_%d" % (row, col), x, y, x + 18, y + 10))
            x += 20.0
        blocks.append((10.0, y, x, y + 10, "line", row, 0))
        y += 14.0
    # get_text("words") returns (x0,y0,x1,y1,text,block_no,line_no,word_no)
    word_tuples = [(x0, y0, x1, y1, text, 0, 0, 0) for text, x0, y0, x1, y1 in words]
    return FakePage(width=200.0, height=120.0, words=word_tuples, blocks=blocks)


def _scanned_page() -> FakePage:
    """No extractable text at all — a pure image scan."""
    return FakePage(width=612.0, height=792.0, words=[], blocks=[])


def _hybrid_page() -> FakePage:
    """A handful of words (e.g. a stamped page number) with negligible
    coverage on an otherwise-scanned page — must NOT be classified digital."""
    words = [(20.0, 770.0, 40.0, 780.0, "12", 0, 0, 0)]
    blocks = [(20.0, 770.0, 40.0, 780.0, "12", 0, 0)]
    return FakePage(width=612.0, height=792.0, words=words, blocks=blocks)


def test_classify_page_digital_when_words_and_coverage_pass():
    kind, word_count, coverage = classify_page(_digital_page(), min_words=20, min_coverage_ratio=0.02)
    assert kind == "digital"
    assert word_count == 30
    assert coverage > 0.02


def test_classify_page_scanned_when_no_text():
    kind, word_count, coverage = classify_page(_scanned_page(), min_words=20, min_coverage_ratio=0.02)
    assert kind == "scanned"
    assert word_count == 0
    assert coverage == 0.0


def test_classify_page_scanned_for_hybrid_thin_text_layer():
    """A few words with near-zero coverage (a baked-in OCR text layer
    remnant on a scanned page) must fall through to real OCR, not be
    trusted as digital just because *some* text is extractable."""
    kind, word_count, coverage = classify_page(_hybrid_page(), min_words=20, min_coverage_ratio=0.02)
    assert kind == "scanned"
    assert word_count == 1
    assert coverage < 0.02


def test_classify_page_handles_extraction_failure_as_scanned():
    class BrokenPage:
        rect = SimpleNamespace(width=100.0, height=100.0)

        def get_text(self, mode):
            raise RuntimeError("boom")

    kind, word_count, coverage = classify_page(BrokenPage(), min_words=1, min_coverage_ratio=0.0)
    assert kind == "scanned"
    assert word_count == 0
    assert coverage == 0.0


def test_extract_native_words_scales_points_to_pixels():
    page = FakePage(
        width=100.0,
        height=100.0,
        words=[(10.0, 20.0, 30.0, 25.0, "Hello", 0, 0, 0), (40.0, 20.0, 60.0, 25.0, "World", 0, 0, 1)],
        blocks=[],
    )
    merged, words = extract_native_words(page, dpi=144, page_index=2)
    # dpi=144 -> scale factor 2.0 from PDF points (72/inch) to pixels.
    assert merged == "Hello World"
    assert len(words) == 2
    first = words[0]
    assert first.bbox.x == 20 and first.bbox.y == 40
    assert first.bbox.w == 40 and first.bbox.h == 10
    assert first.page == 2
    assert first.engines == ["native_pdf_text"]
    assert first.ocr_confidence == 1.0
    assert first.engine_agreement == 1.0


def test_extract_native_words_skips_blank_tokens():
    page = FakePage(width=100.0, height=100.0, words=[(0.0, 0.0, 5.0, 5.0, "   ", 0, 0, 0)], blocks=[])
    merged, words = extract_native_words(page, dpi=72, page_index=0)
    assert merged == ""
    assert words == []


def test_classify_and_extract_returns_scanned_for_none_page():
    kind, merged, words = classify_and_extract(
        None, dpi=200, page_index=0, min_words=20, min_coverage_ratio=0.02
    )
    assert kind == "scanned"
    assert merged == ""
    assert words == []


def test_classify_and_extract_digital_page_end_to_end():
    kind, merged, words = classify_and_extract(
        _digital_page(), dpi=72, page_index=1, min_words=20, min_coverage_ratio=0.02
    )
    assert kind == "digital"
    assert merged
    assert len(words) == 30
    assert all(w.page == 1 for w in words)


def test_classify_and_extract_scanned_page_returns_no_words():
    kind, merged, words = classify_and_extract(
        _scanned_page(), dpi=200, page_index=0, min_words=20, min_coverage_ratio=0.02
    )
    assert kind == "scanned"
    assert merged == ""
    assert words == []
