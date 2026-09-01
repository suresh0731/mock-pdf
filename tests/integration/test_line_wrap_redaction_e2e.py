"""End-to-end test for the line-wrap split wired into
``RedactPipeline._collect_redactions`` (see ``app.pipeline.redact.
_line_wrap_clusters``): a non-tabular match whose own words sit at the
tail of one visual line and the head of the next (a real PDF line break,
not a table cell) must be painted as one tight box per line — with the
mock value on only one of them — instead of a single box spanning nearly
the full width of both lines.

The split is driven by an explicit custom redaction term (rather than a
whole-page NER detector, which this pipeline no longer has) —
``field_detection_enabled``/``dictionary_scan_enabled`` are both off so
this one term is the only detection signal.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.models.pii_chunk import BBox
from app.models.redact import CustomRedactTerm, RedactOptions
from app.pipeline.redact import RedactPipeline
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.ocr.page_renderer import RenderedPage
from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.redact.audit_store import AuditStore
from app.services.redact.ledger_store import LedgerStore

MERGED = "Maria Santos"
NAME_OPTS = RedactOptions(custom_redactions=[CustomRedactTerm(search_value="Maria Santos")])


def _words() -> list[EnsembleWord]:
    """"Maria" at the tail of line 1 (large x), "Santos" at the head of
    line 2 (small x, next row down) — the shape a name split by a real
    PDF line break actually takes."""
    return [
        EnsembleWord(
            text="Maria",
            bbox=BBox(x=500, y=200, w=60, h=18),
            ocr_confidence=0.95,
            engine_agreement=1.0,
            engines=["tesseract"],
            char_start=0,
            char_end=5,
        ),
        EnsembleWord(
            text="Santos",
            bbox=BBox(x=10, y=230, w=70, h=18),
            ocr_confidence=0.95,
            engine_agreement=1.0,
            engines=["tesseract"],
            char_start=6,
            char_end=12,
        ),
    ]


async def _fake_ocr(*args, **kwargs):
    return MERGED, _words(), []


def _pipeline(tmp_path: Path, monkeypatch) -> tuple[RedactPipeline, MockDictionaryStore]:
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("MOCK_DICTIONARY_PATH", str(tmp_path / "mappings.json"))
    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
        field_detection_enabled=False,
        dictionary_scan_enabled=False,
        restrict_to_known_mappings=False,
        _env_file=None,
    )
    mock_store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    pipeline = RedactPipeline(
        settings=settings,
        mock_store=mock_store,
        ledger_store=LedgerStore(base_dir=tmp_path / "shards"),
        audit_store=AuditStore(),
    )
    return pipeline, mock_store


def test_line_wrapped_match_is_split_into_two_tight_boxes(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", _fake_ocr)
    monkeypatch.setattr(
        "app.pipeline.redact.load_pages",
        lambda *a, **k: [RenderedPage(image=Image.new("RGB", (720, 1100), "white"))],
    )
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    pipeline, store = _pipeline(tmp_path, monkeypatch)

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", NAME_OPTS))
    regions = [r for r in audit.redactions if r.entity_type == "CUSTOM"]

    # One tight box per line, not a single box spanning both.
    assert len(regions) == 2
    assert {r.mapping_id for r in regions} == {store.list()[0].mapping_id}

    widths = sorted(r.canonical_bbox.w for r in regions)
    # Neither painted box stretches across the (near-page-width) gap
    # between the two words' visual lines.
    assert widths[0] < 100
    assert widths[1] < 100

    mock_values = [r.mock_value for r in regions]
    # Exactly one box carries the mock value; the other is a blank patch.
    assert sorted(mock_values, key=len)[0] == ""
    assert sorted(mock_values, key=len)[1] != ""


def test_ordinary_left_margin_wrap_is_split_and_spares_neighboring_words(tmp_path, monkeypatch):
    """A plain sentence wrap — not the "opposite ends" extreme the other
    test above uses — must still be split into tight per-line boxes, so
    unrelated words physically sitting in the horizontal gap between the
    two line-fragments (one before the match on line 1, one after it on
    line 2) are never swept into a painted box.

    Modeled on a real regression: "Reksa Dana Pendapatan" tails off
    partway through line 1 (with an unrelated word starting that same
    line, well to its left) and "Bni Am Teakwood" resumes at line 2's
    left margin (with an unrelated word following it). The union of the
    two per-line clusters here is only ~1.19x their combined own widths
    — under the old "extreme ends" 1.6 threshold this fell through as
    one box spanning (and redacting) both unrelated words too.
    """
    merged = "Something Reksa Dana Pendapatan Bni Am Teakwood Else"
    term = "Reksa Dana Pendapatan Bni Am Teakwood"
    something_bbox = BBox(x=50, y=200, w=180, h=18)
    else_bbox = BBox(x=250, y=230, w=50, h=18)

    async def _fake_ocr_left_margin(*args, **kwargs):
        return merged, [
            EnsembleWord(
                text="Something", bbox=something_bbox, ocr_confidence=0.95,
                engine_agreement=1.0, engines=["tesseract"], char_start=0, char_end=9,
            ),
            EnsembleWord(
                text="Reksa", bbox=BBox(x=300, y=200, w=60, h=18), ocr_confidence=0.95,
                engine_agreement=1.0, engines=["tesseract"], char_start=10, char_end=15,
            ),
            EnsembleWord(
                text="Dana", bbox=BBox(x=370, y=200, w=55, h=18), ocr_confidence=0.95,
                engine_agreement=1.0, engines=["tesseract"], char_start=16, char_end=20,
            ),
            EnsembleWord(
                text="Pendapatan", bbox=BBox(x=435, y=200, w=120, h=18), ocr_confidence=0.95,
                engine_agreement=1.0, engines=["tesseract"], char_start=21, char_end=31,
            ),
            EnsembleWord(
                text="Bni", bbox=BBox(x=10, y=230, w=40, h=18), ocr_confidence=0.95,
                engine_agreement=1.0, engines=["tesseract"], char_start=32, char_end=35,
            ),
            EnsembleWord(
                text="Am", bbox=BBox(x=60, y=230, w=35, h=18), ocr_confidence=0.95,
                engine_agreement=1.0, engines=["tesseract"], char_start=36, char_end=38,
            ),
            EnsembleWord(
                text="Teakwood", bbox=BBox(x=105, y=230, w=110, h=18), ocr_confidence=0.95,
                engine_agreement=1.0, engines=["tesseract"], char_start=39, char_end=47,
            ),
            EnsembleWord(
                text="Else", bbox=else_bbox, ocr_confidence=0.95,
                engine_agreement=1.0, engines=["tesseract"], char_start=48, char_end=52,
            ),
        ], []

    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", _fake_ocr_left_margin)
    monkeypatch.setattr(
        "app.pipeline.redact.load_pages",
        lambda *a, **k: [RenderedPage(image=Image.new("RGB", (720, 1100), "white"))],
    )
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    opts = RedactOptions(custom_redactions=[CustomRedactTerm(search_value=term)])
    pipeline, store = _pipeline(tmp_path, monkeypatch)

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", opts))
    regions = [r for r in audit.redactions if r.entity_type == "CUSTOM"]

    # One tight box per line, not a single box spanning both.
    assert len(regions) == 2
    assert {r.mapping_id for r in regions} == {store.list()[0].mapping_id}
    widths = sorted(r.canonical_bbox.w for r in regions)
    assert widths[0] < 260
    assert widths[1] < 260

    def _overlaps(a: BBox, b: BBox) -> bool:
        return a.x < b.x + b.w and a.x + a.w > b.x and a.y < b.y + b.h and a.y + a.h > b.y

    for region in regions:
        assert not _overlaps(region.canonical_bbox, something_bbox)
        assert not _overlaps(region.canonical_bbox, else_bbox)


def test_single_line_match_is_not_split(tmp_path, monkeypatch):
    """Sanity check: an ordinary same-line two-word match still produces
    exactly one painted box, unaffected by the line-wrap logic."""

    async def _fake_ocr_same_line(*args, **kwargs):
        return MERGED, [
            EnsembleWord(
                text="Maria",
                bbox=BBox(x=100, y=200, w=60, h=18),
                ocr_confidence=0.95,
                engine_agreement=1.0,
                engines=["tesseract"],
                char_start=0,
                char_end=5,
            ),
            EnsembleWord(
                text="Santos",
                bbox=BBox(x=165, y=200, w=70, h=18),
                ocr_confidence=0.95,
                engine_agreement=1.0,
                engines=["tesseract"],
                char_start=6,
                char_end=12,
            ),
        ], []

    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", _fake_ocr_same_line)
    monkeypatch.setattr(
        "app.pipeline.redact.load_pages",
        lambda *a, **k: [RenderedPage(image=Image.new("RGB", (720, 1100), "white"))],
    )
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    pipeline, store = _pipeline(tmp_path, monkeypatch)

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", NAME_OPTS))
    regions = [r for r in audit.redactions if r.entity_type == "CUSTOM"]

    assert len(regions) == 1
    assert regions[0].mock_value == store.list()[0].mock_value
