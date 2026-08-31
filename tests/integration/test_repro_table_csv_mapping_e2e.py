"""End-to-end test using the real client mapping CSV against a synthetic
digital PDF that reproduces the user's multi-line table-cell issue.

Verifies that:
- The pipeline imports ``complete_client_mappings.csv`` as the mapping source.
- The synthetic digital table produces bank-name redactions that use the
  curated CSV mock values (not auto-generated labels).
- The second line of the narrow ``Bank`` cell (the word ``Bank`` sitting
  directly below ``Standard Chartered``) is covered by a redaction box, so it
  is not leaked while the same phrase on a single line elsewhere is redacted.

This is a real, dependency-heavy end-to-end run (tens of seconds) and is
marked ``slow`` so it does not slow down the default ``pytest`` run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import fitz
import pytest

from app.config import Settings
from app.models.pii_chunk import BBox
from app.models.redact import RedactOptions
from app.pipeline.redact import RedactPipeline
from app.services.ocr.native_text import classify_and_extract
from app.services.ocr.page_renderer import load_pages
from app.services.pii.mapping_csv import import_mappings_csv
from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.redact.audit_store import AuditStore
from app.services.redact.ledger_store import LedgerStore

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = REPO_ROOT / "repro_table.pdf"
MAPPING_CSV_PATH = REPO_ROOT / "complete_client_mappings.csv"

# Known bank mock values that appear in the CSV for Standard Chartered variants.
_BANK_MOCK_VALUES = {"DSDC_Bank", "CUSTODIAN_A"}


def _bbox_overlap(a: BBox, b: BBox) -> float:
    """Intersection area between two bboxes; 0 if they do not overlap."""
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _bbox_covers(container: BBox, target: BBox, min_fraction: float = 0.5) -> bool:
    """True when ``container`` covers at least ``min_fraction`` of ``target``."""
    overlap = _bbox_overlap(container, target)
    target_area = max(1, target.w * target.h)
    return overlap / target_area >= min_fraction


def _extract_native_word_bboxes(pdf_path: Path, dpi: int = 200) -> dict[str, list[BBox]]:
    """Return native-text word bboxes grouped by text token for the first page."""
    content = pdf_path.read_bytes()
    pages = load_pages(content, pdf_path.name, dpi=dpi)
    raw_page = pages[0]
    page_kind, _merged, words = classify_and_extract(
        raw_page.fitz_page, dpi, 0, min_words=20, min_coverage_ratio=0.02
    )
    assert page_kind == "digital", f"expected digital page, got {page_kind}"
    by_text: dict[str, list[BBox]] = {}
    for w in words:
        by_text.setdefault(w.text, []).append(w.bbox)
    return by_text


def _run_pipeline(tmp_path: Path) -> RedactPipeline:
    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
        field_detection_enabled=True,
        dictionary_scan_enabled=True,
        fuzzy_dictionary_scan_enabled=True,
        restrict_to_known_mappings=False,
        _env_file=None,
    )
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    import_mappings_csv(store, MAPPING_CSV_PATH.read_text(encoding="utf-8"))
    pipeline = RedactPipeline(
        settings=settings,
        mock_store=store,
        ledger_store=LedgerStore(base_dir=tmp_path / "shards"),
        audit_store=AuditStore(),
    )
    return pipeline


@pytest.mark.skipif(not PDF_PATH.exists(), reason="repro_table.pdf not present")
@pytest.mark.skipif(not MAPPING_CSV_PATH.exists(), reason="complete_client_mappings.csv not present")
def test_repro_table_bank_cells_redacted_with_csv_mapping(tmp_path: Path):
    pipeline = _run_pipeline(tmp_path)
    opts = RedactOptions()
    content = PDF_PATH.read_bytes()

    redacted_pdf, audit, _ = asyncio.run(pipeline.run(content, PDF_PATH.name, opts))

    # Basic sanity: we processed a single digital page.
    assert audit.page_count == 1
    assert audit.pages[0].page_kind == "digital"

    # Bank-name redactions must exist and use the curated CSV mock values.
    bank_redactions = [r for r in audit.redactions if r.mock_value in _BANK_MOCK_VALUES]
    assert len(bank_redactions) >= 3, (
        f"expected at least 3 bank redactions from the CSV mapping, got {len(bank_redactions)}"
    )

    # The narrow Bank column has "Standard Chartered" on line 1 and "Bank" on
    # line 2 in each data row. Confirm the trailing "Bank" word is covered by
    # a redaction box so it is not leaked (the original inconsistency report).
    word_bboxes = _extract_native_word_bboxes(PDF_PATH, dpi=opts.dpi)
    narrow_bank_words = [b for b in word_bboxes.get("Bank", []) if b.x < 1500]
    # Exclude the header "Bank" and keep the three data rows.
    data_bank_words = [b for b in narrow_bank_words if b.y > 200]
    assert len(data_bank_words) >= 3, (
        f"expected >= 3 data-row 'Bank' words in narrow Bank column, got {len(data_bank_words)}"
    )

    for bank_word in data_bank_words:
        covered = any(
            _bbox_covers(r.padded_bbox, bank_word, min_fraction=0.5)
            for r in bank_redactions
        )
        assert covered, (
            f"narrow Bank cell 'Bank' word at {bank_word} is not covered by any bank redaction"
        )

    # The redacted PDF should still be a valid PDF with one page.
    doc = fitz.open(stream=redacted_pdf, filetype="pdf")
    assert doc.page_count == 1
    doc.close()
