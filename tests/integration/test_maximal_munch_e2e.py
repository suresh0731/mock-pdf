"""End-to-end test for the maximal-munch window-extension probe wired into
``RedactPipeline._collect_redactions``: a prefix-collision hit (a PII span
covering only "Maksima" when the dictionary already knows "Maksima Plus")
must be resolved by geometrically extending the match to the adjacent word,
not by minting a spurious second dictionary entry for the truncated text.

The truncated span is driven by an explicit custom redaction term (rather
than a whole-page NER detector, which this pipeline no longer has) —
``field_detection_enabled``/``dictionary_scan_enabled`` are both off here so
this one term is the only detection signal, isolating the maximal-munch/
spillover logic under test from dictionary-scan's own (correct, untruncated)
match of "Maksima Plus" once it's a known entry.
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
from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.redact.audit_store import AuditStore
from app.services.redact.ledger_store import LedgerStore

MERGED = "Maksima Plus paid the invoice"
MAKSIMA_START = MERGED.find("Maksima")
MAKSIMA_END = MAKSIMA_START + len("Maksima")

TRUNCATED_TERM_OPTS = RedactOptions(custom_redactions=[CustomRedactTerm(search_value="Maksima")])


def _words() -> list[EnsembleWord]:
    return [
        EnsembleWord(
            text="Maksima",
            bbox=BBox(x=40, y=200, w=90, h=18),
            ocr_confidence=0.95,
            engine_agreement=1.0,
            engines=["tesseract"],
            char_start=MAKSIMA_START,
            char_end=MAKSIMA_END,
        ),
        EnsembleWord(
            text="Plus",
            bbox=BBox(x=140, y=200, w=50, h=18),
            ocr_confidence=0.95,
            engine_agreement=1.0,
            engines=["tesseract"],
            char_start=MAKSIMA_END + 1,
            char_end=MAKSIMA_END + 1 + len("Plus"),
        ),
    ]


async def _fake_ocr(*args, **kwargs):
    return MERGED, _words(), []


def _pipeline(
    tmp_path: Path, monkeypatch, *, spillover_safety_net_enabled: bool = True
) -> tuple[RedactPipeline, MockDictionaryStore]:
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("MOCK_DICTIONARY_PATH", str(tmp_path / "mappings.json"))
    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
        field_detection_enabled=False,
        dictionary_scan_enabled=False,
        restrict_to_known_mappings=False,
        spillover_safety_net_enabled=spillover_safety_net_enabled,
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


def test_truncated_span_extends_to_match_known_longer_entry(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", _fake_ocr)
    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: [Image.new("RGB", (720, 1100), "white")])
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    pipeline, store = _pipeline(tmp_path, monkeypatch)
    known = store.resolve("Maksima Plus")

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", TRUNCATED_TERM_OPTS))
    org_region = next(r for r in audit.redactions if r.entity_type == "CUSTOM")

    assert org_region.mapping_id == known.mapping_id
    assert org_region.mock_value == known.mock_value
    # No spurious second entry was minted for the truncated "Maksima" text.
    assert len(store.list()) == 1
    # The painted region grew to cover "Plus" too, not just "Maksima".
    assert org_region.canonical_bbox.w >= 150


def test_spillover_net_absorbs_orphan_even_without_a_collision(tmp_path, monkeypatch):
    """No dictionary collision (so the maximal-munch probe never fires) —
    but the orphaned "Plus" word is still visually absorbed into the
    "Maksima" redaction's box by the independent spillover safety net,
    even though the dictionary mapping itself is only for "Maksima"."""
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", _fake_ocr)
    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: [Image.new("RGB", (720, 1100), "white")])
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    pipeline, store = _pipeline(tmp_path, monkeypatch)

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", TRUNCATED_TERM_OPTS))
    org_region = next(r for r in audit.redactions if r.entity_type == "CUSTOM")

    assert len(store.list()) == 1
    assert store.list()[0].normalized == "maksima"
    # The dictionary only knows "Maksima", yet the painted box still grew
    # to cover "Plus" too, so no fragment of it is left exposed.
    assert org_region.canonical_bbox.w >= 150


def test_spillover_net_disabled_leaves_orphan_exposed(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", _fake_ocr)
    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: [Image.new("RGB", (720, 1100), "white")])
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    pipeline, store = _pipeline(tmp_path, monkeypatch, spillover_safety_net_enabled=False)

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", TRUNCATED_TERM_OPTS))
    org_region = next(r for r in audit.redactions if r.entity_type == "CUSTOM")

    assert len(store.list()) == 1
    assert store.list()[0].normalized == "maksima"
    assert org_region.canonical_bbox.w < 150
