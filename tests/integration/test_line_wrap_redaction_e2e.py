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
