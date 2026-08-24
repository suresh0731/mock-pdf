"""General-purpose validation script: run the real RedactPipeline against
any sample PDF (optionally trimmed to a page range) with optional custom
redaction terms and/or pre-seeded mock-dictionary entries, then print the
audit summary, every redaction, and the resulting mock-dictionary entries.

Usage (from repo root, with rapidocr installed / OCR_REQUIRED_ENGINES set):
    python scripts/validate_pdf.py <path.pdf> [--pages N] \
        [--term "SEARCH_TEXT=MOCK_VALUE"] [--restrict-known / --no-restrict-known]

Examples:
    # Baseline: only the persistent mapping table, no custom terms.
    python scripts/validate_pdf.py Account-Statement-HDFC-6month.pdf --pages 2

    # With custom terms (the "Extra text to replace" UI field equivalent).
    python scripts/validate_pdf.py Account-Statement-HDFC-6month.pdf --pages 2 \
        --term "NARESH G=MOCK_NARESH" --term "MUNISHAMY GOVINDAPPA=MOCK_GOVINDAPPA"
"""

from __future__ import annotations

import argparse
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
from app.models.redact import CustomRedactTerm, RedactOptions  # noqa: E402
from app.pipeline.redact import RedactPipeline  # noqa: E402
from app.services.ocr.engines import configure_tesseract  # noqa: E402
from app.services.pii.mock_dictionary import MockDictionaryStore  # noqa: E402
from app.services.redact.audit_store import AuditStore  # noqa: E402
from app.services.redact.ledger_store import LedgerStore  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Path to the sample PDF (relative to repo root or absolute)")
    parser.add_argument("--pages", type=int, default=None, help="Only process the first N pages")
    parser.add_argument(
        "--term",
        action="append",
        default=[],
        dest="terms",
        help='Custom redaction term, "SEARCH_TEXT" or "SEARCH_TEXT=MOCK_VALUE" (repeatable)',
    )
    parser.add_argument(
        "--seed-mapping",
        action="append",
        default=[],
        dest="seed_mappings",
        help='Pre-seed the mock dictionary only (no detection effect), "NAME=MOCK_VALUE" (repeatable)',
    )
    parser.add_argument(
        "--restrict-known",
        dest="restrict_known",
        action="store_true",
        default=None,
        help="Force restrict_to_known_mappings=True for field-anchored detection",
    )
    parser.add_argument(
        "--no-restrict-known",
        dest="restrict_known",
        action="store_false",
        help="Force restrict_to_known_mappings=False (allow auto-create)",
    )
    return parser.parse_args()


def _trim_pdf(src: Path, pages: int | None) -> Path:
    if pages is None:
        return src
    import fitz

    doc = fitz.open(src)
    if pages >= doc.page_count:
        return src
    doc.select(list(range(pages)))
    out_path = src.with_name(src.stem + f".first{pages}.pdf")
    doc.save(out_path)
    return out_path


async def main() -> None:
    args = _parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = REPO_ROOT / pdf_path
    if not pdf_path.exists():
        raise SystemExit(f"Sample file not found: {pdf_path}")

    if args.restrict_known is not None:
        os.environ["RESTRICT_TO_KNOWN_MAPPINGS"] = "true" if args.restrict_known else "false"

    settings = get_settings()
    settings.shard_base_path.mkdir(parents=True, exist_ok=True)
    settings.mock_dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    configure_tesseract()

    mock_store = MockDictionaryStore(snapshot_path=settings.mock_dictionary_path)
    for seed in args.seed_mappings:
        name, _, mock_value = seed.partition("=")
        mock_store.upsert(name.strip(), (mock_value or "MOCK_SEED").strip())

    ledger_store = LedgerStore(base_dir=tmp_dir / "shards")
    audit_store = AuditStore()

    pipeline = RedactPipeline(
        settings=settings,
        mock_store=mock_store,
        ledger_store=ledger_store,
        audit_store=audit_store,
    )

    run_path = _trim_pdf(pdf_path, args.pages)
    content = run_path.read_bytes()

    custom_redactions = []
    for term in args.terms:
        search_value, _, mock_value = term.partition("=")
        kwargs = {"search_value": search_value.strip()}
        if mock_value.strip():
            kwargs["mock_label"] = mock_value.strip()
        custom_redactions.append(CustomRedactTerm(**kwargs))

    opts = RedactOptions(custom_redactions=custom_redactions)
    pdf_bytes, audit, session = await pipeline.run(content, run_path.name, opts)

    out_pdf = pdf_path.with_name(pdf_path.stem + ".redacted.pdf")
    out_pdf.write_bytes(pdf_bytes)

    print("=== SETTINGS ===")
    print(f"restrict_to_known_mappings={settings.restrict_to_known_mappings}")
    print(f"field_detection_enabled={settings.field_detection_enabled}")
    print(f"dictionary_scan_enabled={settings.dictionary_scan_enabled}")
    print(f"custom_redactions={[t.search_value for t in custom_redactions]}")
    print()
    print("=== AUDIT SUMMARY ===")
    print(json.dumps(audit.summary, indent=2, default=str))
    print()
    print("=== REDACTIONS ===")
    if not audit.redactions:
        print("(none)")
    for r in audit.redactions:
        print(
            f"page={r.page} entity_type={r.entity_type!r} "
            f"mock_value={r.mock_value!r} mapping_id={r.mapping_id!r} "
            f"assignment_source={r.assignment_source!r}"
        )
    print()
    print("=== MOCK DICTIONARY ENTRIES (name + mock value only) ===")
    for e in mock_store.list():
        print(f"mapping_id={e.mapping_id!r} source_text={e.source_text!r} mock_value={e.mock_value!r}")
    print()
    print(f"Redacted PDF written to: {out_pdf}")
    print(f"Session id: {session.session_id}")
    print(f"Request id: {audit.request_id}")


if __name__ == "__main__":
    asyncio.run(main())
