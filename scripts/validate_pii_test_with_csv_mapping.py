"""Runs the real production RedactPipeline.run() end-to-end on pii-test.pdf,
seeded only from complete_client_mappings.csv (the mapping used in the
server), to validate the actual server code path rather than the
visualize_pipeline_stages.py observation harness.

Usage (from repo root):
    python scripts/validate_pii_test_with_csv_mapping.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
tmp_dir = Path(tempfile.mkdtemp(prefix="pii_validate_"))
os.environ.setdefault("SHARD_BASE_PATH", str(tmp_dir))

from app.config import get_settings  # noqa: E402
from app.models.redact import RedactOptions  # noqa: E402
from app.pipeline.redact import RedactPipeline  # noqa: E402
from app.services.ocr.engines import configure_tesseract  # noqa: E402
from app.services.pii.mapping_csv import import_mappings_csv  # noqa: E402
from app.services.pii.mock_dictionary import MockDictionaryStore  # noqa: E402
from app.services.redact.audit_store import AuditStore  # noqa: E402
from app.services.redact.ledger_store import LedgerStore  # noqa: E402


async def main() -> None:
    settings = get_settings()
    settings.shard_base_path.mkdir(parents=True, exist_ok=True)
    configure_tesseract()

    mock_snapshot = tmp_dir / "mappings.json"
    mock_store = MockDictionaryStore(snapshot_path=mock_snapshot)
    csv_path = REPO_ROOT / "complete_client_mappings.csv"
    result = import_mappings_csv(mock_store, csv_path.read_text(encoding="utf-8"))
    print(f"Imported {csv_path.name}: {result.inserted} inserted, "
          f"{result.skipped_existing} skipped (existing), {result.skipped_invalid} skipped (invalid)")

    ledger_store = LedgerStore(base_dir=tmp_dir / "shards")
    audit_store = AuditStore()
    pipeline = RedactPipeline(
        settings=settings, mock_store=mock_store, ledger_store=ledger_store, audit_store=audit_store
    )

    src = REPO_ROOT / "pii-test.pdf"
    file_bytes = src.read_bytes()
    opts = RedactOptions()

    pdf_bytes, audit, session = await pipeline.run(file_bytes, src.name, opts)

    out_path = REPO_ROOT / "pii-test.redacted.pdf"
    out_path.write_bytes(pdf_bytes)
    print(f"\nWrote {out_path} ({len(pdf_bytes)} bytes)")
    print(f"pages={audit.page_count} redactions={audit.summary['redaction_count']} "
          f"avg_confidence={audit.summary['avg_confidence']}")
    for p in audit.pages:
        print(f"  page={p.page} redactions={p.redaction_count} page_kind={p.page_kind} blur_tier={p.blur_tier}")


asyncio.run(main())
