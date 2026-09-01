"""One-off diagnostic: run the real RedactPipeline against the digital
(native-text) Account-Statement-HDFC-6month-3.pdf to check whether a
known PII value whose own words span two real PDF lines — "near tiles
factory" (line 1) / "mutyyalapete Mulbagal" (line 2), the account
holder's address block — gets tight, per-line redaction boxes (like the
scanned-page line-wrap fix already gives) or one oversized box spanning
both lines, on the *vector-native* digital-PDF redaction path
(app/services/redact/pdf_native_redactor.py).

Usage (from repo root):
    python scripts/diagnose_digital_line_wrap_hdfc.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

tmp_dir = Path(tempfile.mkdtemp(prefix="pii_diagnose_digital_"))
os.environ.setdefault("SHARD_BASE_PATH", str(tmp_dir))
os.environ.setdefault("MOCK_DICTIONARY_PATH", str(tmp_dir / "mappings.json"))

from app.config import get_settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.models.redact import RedactOptions  # noqa: E402
from app.pipeline.redact import RedactPipeline  # noqa: E402
from app.services.ocr.engines import configure_tesseract  # noqa: E402
from app.services.pii.mock_dictionary import MockDictionaryStore  # noqa: E402
from app.services.redact.audit_store import AuditStore  # noqa: E402
from app.services.redact.ledger_store import LedgerStore  # noqa: E402

# "MFAR GREEN HEART, PHASE IV" (line 1) / "MFAR-MANYATA TECH PARK" (line
# 2) — the bank branch address, genuinely contiguous in this page's
# native-text reading order (unlike the customer address block a few
# lines below, which sits beside an unrelated second column and comes
# out interleaved with it — a separate, pre-existing reading-order
# limitation, not the line-wrap behavior this script targets).
ADDRESS_VALUE = "MFAR GREEN HEART, PHASE IV MFAR-MANYATA TECH PARK"
ADDRESS_MOCK = "ADDR_LINE_01"


async def main() -> None:
    configure_logging(level="DEBUG", fmt="plain")

    pdf_path = REPO_ROOT / "Account-Statement-HDFC-6month-3.pdf"
    if not pdf_path.exists():
        raise SystemExit(f"Sample file not found: {pdf_path}")

    settings = get_settings()
    settings.shard_base_path.mkdir(parents=True, exist_ok=True)
    settings.mock_dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    configure_tesseract()

    mock_store = MockDictionaryStore(snapshot_path=settings.mock_dictionary_path)
    known = mock_store.upsert(ADDRESS_VALUE, ADDRESS_MOCK)
    print(f"Seeded mapping: {ADDRESS_VALUE!r} -> {known.mock_value!r} ({known.mapping_id})")

    ledger_store = LedgerStore(base_dir=tmp_dir / "shards")
    audit_store = AuditStore()
    pipeline = RedactPipeline(
        settings=settings, mock_store=mock_store, ledger_store=ledger_store, audit_store=audit_store,
    )

    content = pdf_path.read_bytes()

    print("\n=== Building page states ===")
    page_states = await pipeline._build_page_states(content, pdf_path.name, RedactOptions())
    state = page_states[0]
    print(f"page0 page_kind={state.page_kind} word_count={len(state.ensemble_words)}")

    idx = state.merged_text.lower().find("mfar green heart")
    print(f"\n'mfar green heart' found in merged_text at char {idx}")
    if idx >= 0:
        print(f"Context: ...{state.merged_text[max(0, idx - 30): idx + 80]}...")

    print("\n=== Final RedactionRegions ===")
    all_redactions, page_audits, ledger_rows, brand_zone_dicts = pipeline._collect_redactions(
        [state], RedactOptions(), region_start=0
    )
    for r in all_redactions:
        if r.assignment_source == "brand":
            continue
        print(
            f"  region_id={r.region_id} entity_type={r.entity_type!r} mock_value={r.mock_value!r} "
            f"mapping_id={r.mapping_id!r} canonical_bbox={r.canonical_bbox} padded_bbox={r.padded_bbox}"
        )

    print("\n=== Rendering full pipeline.run() output for visual ground-truth ===")
    out_bytes, audit, session = await pipeline.run(content, pdf_path.name, RedactOptions())
    out_path = REPO_ROOT / "hdfc_digital_line_wrap.redacted.pdf"
    out_path.write_bytes(out_bytes)
    print(f"Wrote {out_path} ({len(out_bytes)} bytes)")
    for r in audit.redactions:
        print(
            f"  page={r.page} entity_type={r.entity_type!r} mock_value={r.mock_value!r} "
            f"mapping_id={r.mapping_id!r} assignment_source={r.assignment_source!r}"
        )


if __name__ == "__main__":
    asyncio.run(main())
