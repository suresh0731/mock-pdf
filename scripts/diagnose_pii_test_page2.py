"""One-off diagnostic: run the real RedactPipeline against pii-test.pdf
page 2 to pin down why "Blife Link Campuran Selaras Plus" and "PT BNI
LIFE INSURANCE" (both table-cell values, both maximal-munch
prefix-collision candidates per complete_client_mappings.csv) end up
with NO redaction painted at all.

Usage (from repo root):
    python scripts/diagnose_pii_test_page2.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

tmp_dir = Path(tempfile.mkdtemp(prefix="pii_diagnose_p2_"))
os.environ.setdefault("SHARD_BASE_PATH", str(tmp_dir))
os.environ.setdefault("MOCK_DICTIONARY_PATH", str(tmp_dir / "mappings.json"))

from app.config import get_settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.pipeline.redact import (  # noqa: E402
    RedactPipeline,
    _build_window_candidates,
    _enclosing_cell_bbox,
    _resolve_maximal_munch_window,
)
from app.models.redact import RedactOptions  # noqa: E402
from app.services.ocr.engines import configure_tesseract  # noqa: E402
from app.services.pii.ensemble_mapper import union_bbox  # noqa: E402
from app.services.pii.field_extractor import extract_field_candidates  # noqa: E402
from app.services.pii.mapping_csv import import_mappings_csv  # noqa: E402
from app.services.pii.mock_dictionary import MockDictionaryStore, normalize_source  # noqa: E402
from app.services.redact.audit_store import AuditStore  # noqa: E402
from app.services.redact.ledger_store import LedgerStore  # noqa: E402


async def main() -> None:
    configure_logging(level="DEBUG", fmt="plain")

    pdf_path = REPO_ROOT / "pii-test.pdf"
    if not pdf_path.exists():
        raise SystemExit(f"Sample file not found: {pdf_path}")

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

    content = pdf_path.read_bytes()

    print("\n=== Building page states ===")
    page_states = await pipeline._build_page_states(content, pdf_path.name, RedactOptions())
    print(f"page_count={len(page_states)}")
    state = page_states[1]
    print(f"page2 page_kind={state.page_kind} word_count={len(state.ensemble_words)} block_count={len(state.blocks)}")

    merged_text = state.merged_text
    for needle, label in [("Selaras", "SELARAS"), ("BNI LIFE INSURANCE", "BNI LIFE INSURANCE"), ("BNI Life", "bni life (mixed case)")]:
        idx = merged_text.upper().find(needle.upper())
        print(f"\n'{needle}' [{label}] found in merged_text at char {idx}")
        if idx >= 0:
            print(f"Context: ...{merged_text[max(0, idx - 60): idx + 80]}...")

    print("\n=== Field-anchored candidates mentioning our targets ===")
    stitch_blocks = state.blocks if settings.docling_cell_stitch_enabled else None
    for cand in extract_field_candidates(merged_text, state.ensemble_words, stitch_blocks, state.word_context):
        text = " ".join(w.text for w in cand.words) if cand.words else merged_text[cand.start:cand.end]
        if "selaras" in text.lower() or "bni life" in text.lower() or "insurance" in text.lower():
            print(f"  entity_type={cand.entity_type!r} score={cand.match_confidence:.3f} "
                  f"chars=({cand.start},{cand.end}) text={text!r} "
                  f"field_role={cand.field_role!r} n_words={len(cand.words) if cand.words else 0}")
            if cand.words:
                for w in cand.words:
                    print(f"      word={w.text!r} bbox={w.bbox}")

                normalized_candidate = normalize_source(text)
                collisions = mock_store.find_prefix_collisions(normalized_candidate)
                print(f"    normalized={normalized_candidate!r}")
                print(f"    find_prefix_collisions -> {[c.source_text for c in collisions]}")

                seed = union_bbox(list(cand.words))
                origin_cell = _enclosing_cell_bbox(seed, state.blocks, word_bboxes=[w.bbox for w in cand.words]) if seed else None
                print(f"    origin_cell={origin_cell}")

                windows = _build_window_candidates(list(cand.words), state.ensemble_words, origin_cell=origin_cell)
                print(f"    window candidates ({len(windows)}):")
                for win in windows:
                    win_text = " ".join(w.text for w in win)
                    win_norm = normalize_source(win_text)
                    win_score = mock_store.best_match_score(win_norm) if win_norm else 0.0
                    print(f"      -> {win_text!r} score={win_score:.3f}")

                chosen_words, chosen_text = _resolve_maximal_munch_window(
                    list(cand.words), state.ensemble_words, mock_store, blocks=state.blocks,
                )
                print(f"    CHOSEN: text={chosen_text!r} n_words={len(chosen_words)}")
                chosen_norm = normalize_source(chosen_text)
                best_id, best_ratio = mock_store._best_match(chosen_norm)
                print(f"    mock_store._best_match({chosen_norm!r}) -> best_id={best_id!r} ratio={best_ratio:.3f}")

    print("\n=== Final RedactionRegions (page2 only) ===")
    all_redactions, page_audits, ledger_rows, brand_zone_dicts = pipeline._collect_redactions(
        [state], RedactOptions(), region_start=0
    )
    for r in all_redactions:
        if r.assignment_source == "brand":
            continue
        print(
            f"  region_id={r.region_id} entity_type={r.entity_type!r} mock_value={r.mock_value!r} "
            f"mapping_id={r.mapping_id!r} assignment_source={r.assignment_source!r} "
            f"canonical_bbox={r.canonical_bbox}"
        )


if __name__ == "__main__":
    asyncio.run(main())
