"""Regression tests: run the *real* pipeline (real OCR engine, real
Docling/img2table structure extraction -- nothing mocked) against every
sample image in `repii/` and assert redaction coverage never drops below an
observed baseline.

Deliberately checks structural coverage (how many redactions of each entity
type were produced) rather than exact OCR text or mock values -- the whole
point of this redesign is that a fixed real image should now produce
identical *output structure* on every machine, but pinning exact substituted
mock strings would just be re-testing the hash function. A drop below the
baseline on a fixed image is a real signal: a bad dependency bump, a broken
extractor change, or a missing/misconfigured OCR engine -- not routine OCR
text drift, which this pipeline no longer has by design (see the
determinism test in tests/regression/test_determinism.py).

These are real, slow, dependency-heavy end-to-end runs (tens of seconds per
image) -- marked `slow` and excluded from the default `pytest` run (see
pytest.ini). Run explicitly with:

    pytest tests/regression/ -m slow
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import pytest

from app.config import Settings
from app.models.redact import RedactAuditResponse, RedactOptions
from app.pipeline.redact import RedactPipeline
from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.redact.audit_store import AuditStore
from app.services.redact.ledger_store import LedgerStore

pytestmark = pytest.mark.slow

REPII_DIR = Path(__file__).resolve().parents[2] / "repii"

# Observed redaction-coverage floor per sample image (entity_type -> minimum
# count, plus "_total"), captured against this pipeline's deterministic
# single-primary-engine (RapidOCR) policy. Update deliberately (with a
# reviewed reason) if a genuine, intentional detection improvement changes
# these numbers -- never just to make a failing run pass.
# BRAND_LOGO removed (2026-08-24): the fixed top-right percent-of-page logo
# zone was dropped in favor of position-agnostic detect_picture_zones()
# (see app/services/pii/brand_zones.py), which covers a logo wherever it
# actually sits rather than only inside one template's assumed corner. Each
# _total below is reduced by exactly 1 to reflect that removed entity —
# every other floor is left untouched from the prior observed baseline.
BASELINE: dict[str, dict[str, int]] = {
    "1000092646.jpg": {"ORGANIZATION": 7, "PERSON": 1, "BRAND_FOOTER": 1, "_total": 9},
    "1000095684.jpg": {"ORGANIZATION": 14, "BRAND_FOOTER": 1, "_total": 15},
    "1000095686.jpg": {"ORGANIZATION": 40, "BRAND_FOOTER": 1, "_total": 41},
    "1000097339.jpg": {"ORGANIZATION": 3, "BRAND_FOOTER": 1, "_total": 4},
    "1000097341.jpg": {"ORGANIZATION": 1, "PERSON": 1, "BRAND_FOOTER": 1, "_total": 3},
    "1000097343.jpg": {"ORGANIZATION": 9, "BRAND_FOOTER": 1, "_total": 10},
    "1000097345.jpg": {"ORGANIZATION": 9, "BRAND_FOOTER": 1, "_total": 10},
}


def _run_pipeline(tmp_path: Path, image_path: Path) -> RedactAuditResponse:
    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
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


@pytest.mark.parametrize("filename", sorted(BASELINE))
def test_repii_sample_meets_redaction_coverage_baseline(tmp_path, filename):
    image_path = REPII_DIR / filename
    if not image_path.exists():
        pytest.skip(f"sample image not present in this checkout: {filename}")

    audit = _run_pipeline(tmp_path, image_path)
    counts = Counter(r.entity_type for r in audit.redactions)
    expected = BASELINE[filename]

    for entity_type, minimum in expected.items():
        if entity_type == "_total":
            continue
        found = counts.get(entity_type, 0)
        assert found >= minimum, (
            f"{filename}: expected >= {minimum} {entity_type} redactions, got {found} "
            f"(full counts: {dict(counts)})"
        )
    assert len(audit.redactions) >= expected["_total"], (
        f"{filename}: expected >= {expected['_total']} total redactions, got {len(audit.redactions)}"
    )
