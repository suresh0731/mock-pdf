"""Determinism self-test: run the same real document through the *real*
pipeline twice (independent temp dirs, independent fresh mock-dictionary
stores -- nothing shared between the two runs) and assert the redaction
output is structurally identical.

This is the "same document, same output, every time, on any machine" claim
this whole redesign is built around -- rather than testing it, this file
demonstrates it directly against a real scanned sample from repii/.

``mapping_id``/``region_id`` are intentionally excluded from the comparison:
``mapping_id`` is a random per-store token (see
``MockDictionaryStore._new_mapping_id``) that is expected to differ between
two independent store instances even for the exact same real value, and
comparing it would defeat the point of this test. ``mock_value`` -- the
actually user-visible, content-hash-derived label -- **is** compared, and is
exactly the piece this redesign made deterministic (see
``deterministic_auto_mock_value``).

Marked `slow` and excluded from the default `pytest` run (see pytest.ini).
Run explicitly with:

    pytest tests/regression/test_determinism.py -m slow
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from app.models.redact import RedactAuditResponse, RedactOptions
from app.pipeline.redact import RedactPipeline
from app.services.ocr.environment_check import check_required_engines
from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.redact.audit_store import AuditStore
from app.services.redact.ledger_store import LedgerStore

pytestmark = pytest.mark.slow

REPII_DIR = Path(__file__).resolve().parents[2] / "repii"
SAMPLE_IMAGE = "1000097341.jpg"


def _run_pipeline(tmp_path: Path, image_path: Path) -> RedactAuditResponse:
    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
        presidio_enabled=False,
        field_detection_enabled=True,
        restrict_to_known_mappings=False,
        _env_file=None,
    )
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    pipeline = RedactPipeline(
        settings=settings,
        mock_store=store,
        ledger_store=LedgerStore(base_dir=tmp_path / "shards"),
        audit_store=AuditStore(),
    )
    _, audit, _ = asyncio.run(pipeline.run(image_path.read_bytes(), image_path.name, RedactOptions()))
    return audit


def _fingerprint(audit: RedactAuditResponse) -> list[tuple]:
    """Redaction shape with every run-specific/random field stripped out --
    see the module docstring for why `mapping_id`/`region_id` don't belong
    in a cross-run determinism comparison."""
    return sorted(
        (
            r.page,
            r.entity_type,
            (r.canonical_bbox.x, r.canonical_bbox.y, r.canonical_bbox.w, r.canonical_bbox.h),
            (r.original_bbox.x, r.original_bbox.y, r.original_bbox.w, r.original_bbox.h),
            (r.padded_bbox.x, r.padded_bbox.y, r.padded_bbox.w, r.padded_bbox.h),
            r.blur_tier,
            r.mock_value,
            r.assignment_source,
        )
        for r in audit.redactions
    )


def test_required_engine_set_is_available_before_trusting_determinism():
    """The determinism claim below only holds if this machine actually has
    the engine set the deterministic OCR policy expects -- assert that
    explicitly instead of silently running the comparison against whatever
    happened to be installed."""
    result = check_required_engines()
    assert result.ok, (
        f"required OCR engines not fully available (required={result.required}, "
        f"available={result.available}, missing={result.missing}, unknown={result.unknown}); "
        "the same-document-twice determinism test below is not meaningful without them"
    )


def test_same_document_twice_produces_identical_redactions(tmp_path):
    image_path = REPII_DIR / SAMPLE_IMAGE
    if not image_path.exists():
        pytest.skip(f"sample image not present in this checkout: {SAMPLE_IMAGE}")

    audit1 = _run_pipeline(tmp_path / "run1", image_path)
    audit2 = _run_pipeline(tmp_path / "run2", image_path)

    assert len(audit1.redactions) == len(audit2.redactions)
    assert _fingerprint(audit1) == _fingerprint(audit2)
    assert audit1.page_count == audit2.page_count
