"""One-off diagnostic: run the real RedactPipeline against
test-input/1000099815.jpg with the CSV mapping from
complete_client_mappings.csv, DEBUG logging enabled, and the OCR output
dump turned on, to pin down exactly which candidate produces the
oversized/garbled region around "PT BANK SMBC INDONESIA TBK ... REKSA
DANA PENDAPATAN TETAP BNI AM TEAKWOOD KELAS R1" and why "SMBC Indonesia"
never gets its own tight redaction there.

Usage (from repo root):
    python scripts/diagnose_line_wrap_smbc.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

tmp_dir = Path(tempfile.mkdtemp(prefix="pii_diagnose_"))
os.environ.setdefault("SHARD_BASE_PATH", str(tmp_dir))
os.environ.setdefault("MOCK_DICTIONARY_PATH", str(tmp_dir / "mappings.json"))
os.environ.setdefault("OCR_OUTPUT_DUMP_ENABLED", "true")
os.environ.setdefault("OCR_OUTPUT_DUMP_DIR", str(tmp_dir / "ocr_dumps"))

from app.config import get_settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.pipeline.page_state import PageProcessState  # noqa: E402
from app.pipeline.redact import RedactPipeline, _SpanCandidate  # noqa: E402
from app.models.redact import RedactOptions  # noqa: E402
from app.services.ocr.engines import configure_tesseract  # noqa: E402
from app.services.pii.custom_redact import find_fuzzy_term_spans, find_term_spans  # noqa: E402
from app.services.pii.ensemble_mapper import map_span_to_ensemble_bbox, union_bbox  # noqa: E402
from app.services.pii.field_extractor import extract_field_candidates  # noqa: E402
from app.services.pii.mapping_csv import import_mappings_csv  # noqa: E402
from app.services.pii.mock_dictionary import MockDictionaryStore, normalize_source  # noqa: E402
from app.services.pii.name_matcher import token_sort_ratio  # noqa: E402
from app.services.redact.audit_store import AuditStore  # noqa: E402
from app.services.redact.ledger_store import LedgerStore  # noqa: E402


def _gather_candidate_spans(pipeline: RedactPipeline, state: PageProcessState, opts: RedactOptions):
    """Mirrors RedactPipeline._collect_redactions's span-gathering half, tagged by source."""
    merged_text = state.merged_text
    ensemble_words = state.ensemble_words
    settings = pipeline.settings
    spans: list[tuple[_SpanCandidate, str]] = []

    if settings.field_detection_enabled:
        stitch_blocks = state.blocks if settings.docling_cell_stitch_enabled else None
        for cand in extract_field_candidates(merged_text, ensemble_words, stitch_blocks, state.word_context):
            spans.append(
                (
                    _SpanCandidate(
                        cand.start, cand.end, cand.entity_type, cand.match_confidence,
                        None, cand.field_role, cand.account_number, cand.words,
                    ),
                    "field",
                )
            )

    if settings.dictionary_scan_enabled:
        claimed_char_ranges = [(s.start, s.end) for s, _src in spans if not s.words]
        for entry in sorted(pipeline.mock_store.list(), key=lambda e: len(e.source_text), reverse=True):
            for start, end in find_term_spans(merged_text, entry.source_text):
                if any(start < ce and end > cs for cs, ce in claimed_char_ranges):
                    continue
                claimed_char_ranges.append((start, end))
                spans.append((_SpanCandidate(start, end, "KNOWN_TERM", 0.95, None, None, None), "dictionary"))

        if settings.fuzzy_dictionary_scan_enabled:
            from app.pipeline.redact import _mask_claimed_ranges

            masked_text = _mask_claimed_ranges(merged_text, claimed_char_ranges)
            fuzzy_candidates: list[tuple[float, int, int]] = []
            for entry in pipeline.mock_store.list():
                for start, end in find_fuzzy_term_spans(
                    masked_text, entry.source_text, threshold=settings.fuzzy_dictionary_scan_threshold
                ):
                    fuzzy_score = token_sort_ratio(
                        normalize_source(entry.source_text), normalize_source(merged_text[start:end])
                    )
                    fuzzy_candidates.append((fuzzy_score, start, end))
                    spans.append(
                        (
                            _SpanCandidate(start, end, "KNOWN_TERM_FUZZY", fuzzy_score, None, None, None),
                            f"dictionary_fuzzy[entry={entry.source_text!r}]",
                        )
                    )

    return spans


async def main() -> None:
    configure_logging(level="DEBUG", fmt="plain")

    img_path = REPO_ROOT / "test-input" / "1000099815.jpg"
    if not img_path.exists():
        raise SystemExit(f"Sample file not found: {img_path}")

    settings = get_settings()
    settings.shard_base_path.mkdir(parents=True, exist_ok=True)
    settings.mock_dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    configure_tesseract()

    mock_store = MockDictionaryStore(snapshot_path=settings.mock_dictionary_path)
    csv_path = REPO_ROOT / "complete_client_mappings.csv"
    result = import_mappings_csv(mock_store, csv_path.read_text(encoding="utf-8"))
    print(f"Imported {csv_path.name}: {result.inserted} inserted, "
          f"{result.skipped_existing} skipped (existing), {result.skipped_invalid} skipped (invalid)")

    ledger_store = LedgerStore(base_dir=tmp_dir / "shards")
    audit_store = AuditStore()
    pipeline = RedactPipeline(
        settings=settings, mock_store=mock_store, ledger_store=ledger_store, audit_store=audit_store,
    )

    content = img_path.read_bytes()

    print("\n=== Building page state ===")
    page_states = await pipeline._build_page_states(content, img_path.name, RedactOptions())
    state = page_states[0]
    print(f"page_kind={state.page_kind} word_count={len(state.ensemble_words)} block_count={len(state.blocks)}")

    merged_text = state.merged_text
    print("\n=== merged_text ===")
    print(merged_text)

    idx = merged_text.upper().find("SMBC")
    print(f"\n'SMBC' found in merged_text at char {idx}")
    if idx >= 0:
        print(f"Context: ...{merged_text[max(0, idx - 60): idx + 120]}...")

    idx2 = merged_text.upper().find("TEAKWOOD")
    print(f"\n'TEAKWOOD' found in merged_text at char {idx2}")
    if idx2 >= 0:
        print(f"Context: ...{merged_text[max(0, idx2 - 120): idx2 + 60]}...")

    from app.services.pii.ensemble_mapper import words_for_span

    for label, (dbg_start, dbg_end) in {
        "SMBC INDONESIA": (450, 464),
        "REKSA DANA...KELAS R1": (517, 569),
    }.items():
        print(f"\n=== words_for_span({dbg_start}, {dbg_end}) [{label}] ===")
        matched = words_for_span(dbg_start, dbg_end, state.ensemble_words)
        for w in matched:
            print(
                f"  text={w.text!r} char_start={w.char_start} char_end={w.char_end} "
                f"bbox=x={w.bbox.x} y={w.bbox.y} w={w.bbox.w} h={w.bbox.h}"
            )

    print("\n=== All ensemble words with char_start/char_end/bbox (sorted by char_start) ===")
    for w in sorted(state.ensemble_words, key=lambda w: w.char_start):
        print(
            f"  char=({w.char_start},{w.char_end}) text={w.text!r} "
            f"bbox=x={w.bbox.x} y={w.bbox.y} w={w.bbox.w} h={w.bbox.h}"
        )

    print("\n=== Raw candidate spans (pre mock-resolution) ===")
    tagged_spans = _gather_candidate_spans(pipeline, state, RedactOptions())
    for span, source in sorted(tagged_spans, key=lambda t: t[0].start):
        if span.words:
            bbox = union_bbox(list(span.words))
            text = " ".join(w.text for w in span.words)
        else:
            bbox = map_span_to_ensemble_bbox(span.start, span.end, state.ensemble_words, merged_text)
            text = merged_text[span.start:span.end]
        print(
            f"  [{source}] entity_type={span.entity_type!r} score={span.score:.3f} "
            f"chars=({span.start},{span.end}) text={text!r} bbox={bbox}"
        )

    print("\n=== Final RedactionRegions (after dedup/mock-resolution/padding) ===")
    all_redactions, page_audits, ledger_rows, brand_zone_dicts = pipeline._collect_redactions(
        [state], RedactOptions(), region_start=0
    )
    for r in all_redactions:
        if r.assignment_source == "brand":
            continue
        print(
            f"  region_id={r.region_id} entity_type={r.entity_type!r} mock_value={r.mock_value!r} "
            f"mapping_id={r.mapping_id!r} assignment_source={r.assignment_source!r} "
            f"canonical_bbox={r.canonical_bbox} padded_bbox={r.padded_bbox}"
        )

    print("\n=== Ledger rows (source_text -> mock_value actually painted) ===")
    for row in ledger_rows:
        print(f"  {row['source_text']!r} -> {row['mock_value']!r} (entity_type={row['entity_type']!r})")

    print("\n=== Rendering full pipeline.run() output for visual ground-truth ===")
    out_bytes, audit, session = await pipeline.run(content, img_path.name, RedactOptions())
    out_path = tmp_dir / "diagnose_current_output.pdf"
    out_path.write_bytes(out_bytes)
    print(f"Wrote {out_path} ({len(out_bytes)} bytes)")
    for r in audit.redactions:
        print(
            f"  page={r.page} entity_type={r.entity_type!r} mock_value={r.mock_value!r} "
            f"mapping_id={r.mapping_id!r} assignment_source={r.assignment_source!r}"
        )


if __name__ == "__main__":
    asyncio.run(main())
