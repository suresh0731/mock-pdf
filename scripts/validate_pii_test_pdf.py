"""One-off validation script: run the real RedactPipeline against
pii-test.pdf using live Tesseract OCR, then print the audit summary and
mock-dictionary entries so the name+mock-value-only mapping simplification
can be checked against a real scanned document.

Usage (from repo root, with TESSERACT_CMD/OCR_REQUIRED_ENGINES set):
    python scripts/validate_pii_test_pdf.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

tmp_dir = Path(tempfile.mkdtemp(prefix="pii_validate_"))
os.environ.setdefault("SHARD_BASE_PATH", str(tmp_dir))
os.environ.setdefault("MOCK_DICTIONARY_PATH", str(tmp_dir / "mappings.json"))

from app.config import get_settings  # noqa: E402
from app.models.redact import RedactOptions  # noqa: E402
from app.pipeline.redact import RedactPipeline  # noqa: E402
from app.services.ocr.engines import configure_tesseract  # noqa: E402
from app.services.pii.mock_dictionary import MockDictionaryStore  # noqa: E402
from app.services.redact.audit_store import AuditStore  # noqa: E402
from app.services.redact.ledger_store import LedgerStore  # noqa: E402


async def main() -> None:
    pdf_path = REPO_ROOT / "pii-test.pdf"
    if not pdf_path.exists():
        raise SystemExit(f"Sample file not found: {pdf_path}")

    settings = get_settings()
    settings.shard_base_path.mkdir(parents=True, exist_ok=True)
    settings.mock_dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    configure_tesseract()

    mock_store = MockDictionaryStore(snapshot_path=settings.mock_dictionary_path)
    ledger_store = LedgerStore(base_dir=tmp_dir / "shards")
    audit_store = AuditStore()

    pipeline = RedactPipeline(
        settings=settings,
        mock_store=mock_store,
        ledger_store=ledger_store,
        audit_store=audit_store,
    )

    content = pdf_path.read_bytes()
    pdf_bytes, audit, session = await pipeline.run(content, pdf_path.name, RedactOptions())

    out_pdf = REPO_ROOT / "pii-test.redacted.pdf"
    out_pdf.write_bytes(pdf_bytes)

    print("=== AUDIT SUMMARY ===")
    print(json.dumps(audit.summary, indent=2, default=str))
    print()
    print("=== REDACTIONS ===")
    for r in audit.redactions:
        print(
            f"page={r.page} entity_type={r.entity_type!r} "
            f"mock_value={r.mock_value!r} mapping_id={r.mapping_id!r} "
            f"assignment_source={r.assignment_source!r} blur_tier={r.blur_tier!r}"
        )
    print()
    print("=== MOCK DICTIONARY ENTRIES (name + mock value only) ===")
    for e in mock_store.list():
        print(f"mapping_id={e.mapping_id!r} source_text={e.source_text!r} mock_value={e.mock_value!r}")
        assert not hasattr(e, "entity_type")
        assert not hasattr(e, "account_number")
        assert not hasattr(e, "field_role")
    print()
    print(f"Redacted PDF written to: {out_pdf}")
    print(f"Session id: {session.session_id}")
    print(f"Request id: {audit.request_id}")


if __name__ == "__main__":
    asyncio.run(main())
