"""Render one numbered image per pipeline stage for a single page, so the
end-to-end flow (ingest -> preprocess -> structure -> OCR -> PII candidate
detection -> mock resolution/padding -> brand zones -> final redaction) can
be inspected visually, stage by stage, to spot exactly where detection or
geometry goes wrong.

This intentionally re-derives each intermediate artifact by calling the
same functions ``RedactPipeline`` calls internally (``_build_page_states``,
the candidate-gathering loop inside ``_collect_redactions``, and
``_collect_redactions`` itself) rather than adding any instrumentation to
the pipeline — it never changes production behavior, it only observes it.

Usage (from repo root):
    python scripts/visualize_pipeline_stages.py pii-test.pdf --page 2

Writes ``1.png`` .. ``9.png`` into ``--out-dir`` (default
``pipeline_stages``), each with a title banner naming the stage and the
counts that matter for spotting a gap (0 structure blocks, 0 candidates,
a candidate that never became a redaction, etc).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

tmp_dir = Path(tempfile.mkdtemp(prefix="pii_visualize_"))
os.environ.setdefault("SHARD_BASE_PATH", str(tmp_dir))

import fitz  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.models.redact import RedactOptions  # noqa: E402
from app.pipeline.page_state import PageProcessState  # noqa: E402
from app.pipeline.redact import RedactPipeline, _mask_claimed_ranges, _SpanCandidate  # noqa: E402
from app.services.ocr.engines import configure_tesseract  # noqa: E402
from app.services.ocr.native_text import classify_and_extract  # noqa: E402
from app.services.ocr.page_renderer import load_pages  # noqa: E402
from app.services.pii.custom_redact import find_fuzzy_term_spans, find_term_spans  # noqa: E402
from app.services.pii.ensemble_mapper import (  # noqa: E402
    map_span_to_ensemble_bbox,
    union_bbox,
)
from app.services.pii.field_extractor import extract_field_candidates  # noqa: E402
from app.services.pii.mock_dictionary import MockDictionaryStore, normalize_source  # noqa: E402
from app.services.pii.name_matcher import token_sort_ratio  # noqa: E402
from app.services.redact.audit_store import AuditStore  # noqa: E402
from app.services.redact.ledger_store import LedgerStore  # noqa: E402
from app.services.redact.pdf_renderer import _draw_on_image  # noqa: E402
from app.services.structure.docling_adapter import extract_structure  # noqa: E402
from app.services.structure.spatial_join import join_words_to_blocks  # noqa: E402


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------

_TTF_CANDIDATES = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_TTF_BOLD_CANDIDATES = (
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    "DejaVuSans-Bold.ttf",
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for candidate in _TTF_BOLD_CANDIDATES if bold else _TTF_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


_TITLE_FONT = _font(30, bold=True)
_SUB_FONT = _font(18)
_LABEL_FONT = _font(15, bold=True)
_LEGEND_FONT = _font(16)

BLOCK_COLORS: dict[str, tuple[int, int, int]] = {
    "table": (0, 90, 200),
    "cell": (0, 170, 220),
    "paragraph": (140, 140, 140),
    "header": (230, 140, 0),
    "footer": (150, 90, 40),
    "picture": (160, 0, 160),
    "label": (0, 150, 90),
}

SPAN_SOURCE_COLORS: dict[str, tuple[int, int, int]] = {
    "field": (30, 100, 220),
    "custom": (220, 30, 30),
    "dictionary": (30, 160, 60),
    "dictionary_fuzzy": (230, 160, 20),
}

_ENTITY_PALETTE = (
    (200, 30, 30),
    (30, 110, 200),
    (30, 150, 60),
    (200, 120, 0),
    (140, 40, 180),
    (0, 150, 150),
    (170, 90, 40),
)


def _entity_color(entity_type: str) -> tuple[int, int, int]:
    idx = sum(ord(c) for c in entity_type) % len(_ENTITY_PALETTE)
    return _ENTITY_PALETTE[idx]


def _title_banner(width: int, title: str, subtitle: str = "") -> Image.Image:
    height = 96 if subtitle else 64
    banner = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(banner)
    draw.text((24, 14), title, fill=(15, 15, 15), font=_TITLE_FONT)
    if subtitle:
        draw.text((24, 56), subtitle, fill=(80, 80, 80), font=_SUB_FONT)
    draw.line([(0, height - 1), (width, height - 1)], fill=(210, 210, 210), width=2)
    return banner


def _with_title(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    banner = _title_banner(image.width, title, subtitle)
    out = Image.new("RGB", (image.width, image.height + banner.height), (255, 255, 255))
    out.paste(banner, (0, 0))
    out.paste(image.convert("RGB"), (0, banner.height))
    return out


def _label_tag(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    """Small filled tag with white text, anchored just above (x, y)."""
    tw = draw.textlength(text, font=_LABEL_FONT)
    pad = 3
    ty = max(0, y - 20)
    draw.rectangle([x, ty, x + tw + 2 * pad, ty + 18], fill=color)
    draw.text((x + pad, ty + 1), text, fill=(255, 255, 255), font=_LABEL_FONT)


def _draw_legend(image: Image.Image, entries: list[tuple[str, tuple[int, int, int]]]) -> None:
    if not entries:
        return
    draw = ImageDraw.Draw(image)
    pad = 10
    swatch = 16
    line_h = 26
    box_w = max(draw.textlength(label, font=_LEGEND_FONT) for label, _ in entries) + swatch + 3 * pad
    box_h = line_h * len(entries) + pad
    x0, y0 = image.width - int(box_w) - 16, 16
    draw.rectangle([x0, y0, x0 + int(box_w), y0 + box_h], fill=(255, 255, 255), outline=(120, 120, 120), width=2)
    for i, (label, color) in enumerate(entries):
        ly = y0 + pad + i * line_h
        draw.rectangle([x0 + pad, ly, x0 + pad + swatch, ly + swatch], fill=color)
        draw.text((x0 + pad + swatch + 8, ly - 1), label, fill=(20, 20, 20), font=_LEGEND_FONT)


def _save(image: Image.Image, out_dir: Path, index: int, label: str) -> None:
    path = out_dir / f"{index}.png"
    image.convert("RGB").save(path)
    print(f"  [{index}] {label} -> {path.name}")


# --------------------------------------------------------------------------
# Pipeline driver
# --------------------------------------------------------------------------


def _extract_single_page(src: Path, page_number: int) -> Path:
    """1-indexed ``page_number`` -> a temp single-page PDF (avoids the OCR
    concurrency path entirely, since only one page ever exists to process).
    """
    doc = fitz.open(src)
    page_index = page_number - 1
    if not (0 <= page_index < doc.page_count):
        raise SystemExit(
            f"--page {page_number} out of range (document has {doc.page_count} pages)"
        )
    doc.select([page_index])
    out_path = tmp_dir / f"{src.stem}.page{page_number}.pdf"
    doc.save(out_path)
    return out_path


def _gather_candidate_spans(
    pipeline: RedactPipeline,
    state: PageProcessState,
    opts: RedactOptions,
) -> list[tuple[_SpanCandidate, str]]:
    """Field-anchored / custom-term / dictionary-scan spans, before mock
    resolution, dedup, or padding — mirrors the first half of
    ``RedactPipeline._collect_redactions`` exactly, tagging each with its
    detection source for the stage-5 visualization.
    """
    merged_text = state.merged_text
    ensemble_words = state.ensemble_words
    settings = pipeline.settings
    spans: list[tuple[_SpanCandidate, str]] = []

    if settings.field_detection_enabled:
        stitch_blocks = state.blocks if settings.docling_cell_stitch_enabled else None
        for cand in extract_field_candidates(
            merged_text, ensemble_words, stitch_blocks, state.word_context
        ):
            spans.append(
                (
                    _SpanCandidate(
                        cand.start,
                        cand.end,
                        cand.entity_type,
                        cand.match_confidence,
                        None,
                        cand.field_role,
                        cand.account_number,
                        cand.words,
                    ),
                    "field",
                )
            )

    for term in opts.custom_redactions:
        for start, end in find_term_spans(merged_text, term.search_value):
            spans.append((_SpanCandidate(start, end, "CUSTOM", 0.95, term, None, None), "custom"))

    if settings.dictionary_scan_enabled:
        claimed_char_ranges = [(s.start, s.end) for s, _src in spans if not s.words]
        for entry in sorted(pipeline.mock_store.list(), key=lambda e: len(e.source_text), reverse=True):
            for start, end in find_term_spans(merged_text, entry.source_text):
                if any(start < ce and end > cs for cs, ce in claimed_char_ranges):
                    continue
                claimed_char_ranges.append((start, end))
                spans.append((_SpanCandidate(start, end, "KNOWN_TERM", 0.95, None, None, None), "dictionary"))

        if settings.fuzzy_dictionary_scan_enabled:
            # Probed for every entry (not only ones with zero exact hits)
            # against a masked copy of merged_text — see
            # RedactPipeline._collect_redactions / _mask_claimed_ranges: an
            # entry can appear both cleanly (already exact-matched
            # elsewhere) and garbled in more than one other spot, and a
            # clean occurrence would otherwise win every slot in
            # find_fuzzy_term_spans's own per-entry match budget before a
            # different, still-uncaught garbled one got a turn.
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

            # Best-scoring fuzzy match wins a contested range — see the
            # matching comment in RedactPipeline._collect_redactions for why
            # length can't decide priority here the way it safely does for
            # exact matches.
            for fuzzy_score, start, end in sorted(fuzzy_candidates, key=lambda c: c[0], reverse=True):
                if any(start < ce and end > cs for cs, ce in claimed_char_ranges):
                    continue
                claimed_char_ranges.append((start, end))
                spans.append(
                    (
                        _SpanCandidate(start, end, "KNOWN_TERM_FUZZY", fuzzy_score, None, None, None),
                        "dictionary_fuzzy",
                    )
                )

    return spans


def _span_bbox(span: _SpanCandidate, ensemble_words, merged_text: str):
    if span.words:
        return union_bbox(list(span.words)), " ".join(w.text for w in span.words)
    bbox = map_span_to_ensemble_bbox(span.start, span.end, ensemble_words, merged_text)
    return bbox, merged_text[span.start : span.end]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", default="pii-test.pdf", help="Path to the sample PDF")
    parser.add_argument("--page", type=int, default=2, help="1-indexed page number to visualize (default: 2)")
    parser.add_argument(
        "--out-dir", default="pipeline_stages", help="Output folder for the numbered images"
    )
    parser.add_argument(
        "--use-real-dictionary",
        dest="use_real_dictionary",
        action="store_true",
        default=True,
        help="Seed the mock dictionary from the real data/mock-dictionary/mappings.json (read-only copy)",
    )
    parser.add_argument(
        "--empty-dictionary",
        dest="use_real_dictionary",
        action="store_false",
        help="Start from an empty mock dictionary instead",
    )
    parser.add_argument(
        "--mapping-csv",
        default=None,
        help="Additional source_text,mock_value CSV to import on top of the "
        "base dictionary (same shape as the mock-dictionary export/template) "
        "- lets a doc-specific test mapping drive a visualization run "
        "without touching the real data/mock-dictionary/mappings.json",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = REPO_ROOT / pdf_path
    if not pdf_path.exists():
        raise SystemExit(f"Sample file not found: {pdf_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    settings.shard_base_path.mkdir(parents=True, exist_ok=True)

    mock_snapshot = tmp_dir / "mappings.json"
    if args.use_real_dictionary and settings.mock_dictionary_path.exists():
        shutil.copy(settings.mock_dictionary_path, mock_snapshot)
        print(f"Seeded mock dictionary from {settings.mock_dictionary_path} (read-only copy)")
    settings.mock_dictionary_path = mock_snapshot
    configure_tesseract()

    mock_store = MockDictionaryStore(snapshot_path=mock_snapshot)
    if args.mapping_csv:
        from app.services.pii.mapping_csv import import_mappings_csv

        csv_path = Path(args.mapping_csv)
        if not csv_path.is_absolute():
            csv_path = REPO_ROOT / csv_path
        result = import_mappings_csv(mock_store, csv_path.read_text(encoding="utf-8"))
        print(
            f"Imported {args.mapping_csv}: {result.inserted} inserted, "
            f"{result.skipped_existing} skipped (existing), {result.skipped_invalid} skipped (invalid)"
        )
    ledger_store = LedgerStore(base_dir=tmp_dir / "shards")
    audit_store = AuditStore()
    pipeline = RedactPipeline(
        settings=settings, mock_store=mock_store, ledger_store=ledger_store, audit_store=audit_store
    )

    run_path = _extract_single_page(pdf_path, args.page)
    content = run_path.read_bytes()
    opts = RedactOptions()

    print(f"\n=== Visualizing {pdf_path.name} page {args.page} -> {out_dir} ===\n")

    # ---- Stage 1: raw rendered page --------------------------------------
    pages = load_pages(content, run_path.name, dpi=opts.dpi)
    raw_page = pages[0]
    _save(
        _with_title(raw_page.image, f"1. Raw Input — Rendered Page ({args.page} of {pdf_path.name})",
                    f"size={raw_page.image.width}x{raw_page.image.height} dpi={opts.dpi}"),
        out_dir, 1, "raw rendered page",
    )

    # ---- Native-text classification + preprocessing ----------------------
    if settings.native_text_bypass_enabled:
        page_kind, native_merged, native_words = classify_and_extract(
            raw_page.fitz_page, opts.dpi, 0, settings.native_text_min_words, settings.native_text_min_coverage_pct
        )
    else:
        page_kind, native_merged, native_words = "scanned", "", []

    from app.services.preprocess.canonical import canonicalize_page

    canonical = canonicalize_page(
        raw_page.image, 0,
        strip_gridlines=settings.strip_gridlines_enabled,
        deskew=settings.deskew_enabled and page_kind != "digital",
    )
    t = canonical.transform
    _save(
        _with_title(
            canonical.canonical_image,
            "2. Preprocessed / Canonical Image (what OCR actually reads)",
            f"page_kind={page_kind} blur_tier={t.blur_tier} (var={t.blur_variance:.1f}) "
            f"skew_deg={t.skew_angle_deg} preprocessing={t.preprocessing} "
            f"gridlines_stripped={settings.strip_gridlines_enabled}",
        ),
        out_dir, 2, "canonical/preprocessed image",
    )

    # ---- Stage 3: structure extraction -----------------------------------
    blocks = extract_structure(canonical.original_image)
    if settings.img2table_enabled:
        try:
            from app.services.structure.table_geometry import extract_table_geometry, merge_table_geometry

            img2table_blocks = extract_table_geometry(canonical.original_image)
            blocks = merge_table_geometry(blocks, img2table_blocks, min_table_iou=settings.img2table_min_table_iou)
        except Exception:
            print("  (img2table merge unavailable/failed, keeping Docling-only blocks)")

    struct_img = canonical.original_image.convert("RGB").copy()
    draw = ImageDraw.Draw(struct_img)
    type_counts: dict[str, int] = {}
    for block in blocks:
        color = BLOCK_COLORS.get(block.block_type, (90, 90, 90))
        b = block.bbox
        draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=color, width=3)
        type_counts[block.block_type] = type_counts.get(block.block_type, 0) + 1
    legend = [(f"{k} ({v})", BLOCK_COLORS.get(k, (90, 90, 90))) for k, v in sorted(type_counts.items())]
    if not legend:
        legend = [("no blocks detected — structure degraded", (200, 30, 30))]
    _draw_legend(struct_img, legend)
    gap_note = "" if blocks else "GAP: 0 structure blocks (Docling/img2table unavailable) — no table/cell geometry downstream"
    _save(
        _with_title(struct_img, "3. Structure Extraction — Docling/img2table blocks", gap_note),
        out_dir, 3, "structure blocks",
    )

    # ---- Stage 4: OCR ensemble words -------------------------------------
    from app.services.locale.resolver import resolve_languages
    from app.services.ocr.ensemble import ensemble_ocr_page

    table_regions = [b.bbox for b in blocks if b.block_type in ("table", "cell")]
    locale, langs, tess_lang = resolve_languages(opts.locale, opts.languages, opts.auto_detect, "")

    if page_kind == "digital":
        merged_text, ensemble_words = native_merged, native_words
    else:
        merged_text, ensemble_words, _ = await ensemble_ocr_page(
            canonical.canonical_image, canonical.page_index, tess_lang, langs,
            table_regions=table_regions, engine_filter=opts.ocr_engines,
        )

    ocr_img = canonical.canonical_image.convert("RGB").copy()
    draw = ImageDraw.Draw(ocr_img)
    single_engine = 0
    for w in ensemble_words:
        color = (30, 160, 60) if len(w.engines) > 1 else (220, 150, 0)
        if len(w.engines) <= 1:
            single_engine += 1
        b = w.bbox
        draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=color, width=2)
    _draw_legend(
        ocr_img,
        [("multi-engine agreement", (30, 160, 60)), ("single-engine only", (220, 150, 0))],
    )
    _save(
        _with_title(
            ocr_img,
            f"4. OCR Ensemble — {len(ensemble_words)} words detected (page_kind={page_kind})",
            f"single-engine words={single_engine} (lower-confidence gap candidates)",
        ),
        out_dir, 4, "OCR word boxes",
    )

    word_context = join_words_to_blocks(ensemble_words, blocks)
    state = PageProcessState(
        canonical=canonical, merged_text=merged_text, ensemble_words=ensemble_words,
        blocks=blocks, word_context=word_context, page_kind=page_kind,
    )

    # ---- Stage 5: raw PII candidates (before mock resolution) ------------
    tagged_spans = _gather_candidate_spans(pipeline, state, opts)
    cand_img = canonical.original_image.convert("RGB").copy()
    draw = ImageDraw.Draw(cand_img)
    source_counts: dict[str, int] = {}
    for span, source in tagged_spans:
        bbox, text = _span_bbox(span, ensemble_words, merged_text)
        if bbox is None or not text.strip():
            continue
        color = SPAN_SOURCE_COLORS[source]
        source_counts[source] = source_counts.get(source, 0) + 1
        draw.rectangle([bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h], outline=color, width=3)
        _label_tag(draw, bbox.x, bbox.y, span.entity_type, color)
    _draw_legend(
        cand_img,
        [
            (f"field-anchored ({source_counts.get('field', 0)})", SPAN_SOURCE_COLORS["field"]),
            (f"dictionary-scan ({source_counts.get('dictionary', 0)})", SPAN_SOURCE_COLORS["dictionary"]),
            (
                f"dictionary-scan fuzzy ({source_counts.get('dictionary_fuzzy', 0)})",
                SPAN_SOURCE_COLORS["dictionary_fuzzy"],
            ),
            (f"custom-term ({source_counts.get('custom', 0)})", SPAN_SOURCE_COLORS["custom"]),
        ],
    )
    _save(
        _with_title(
            cand_img,
            f"5. PII Candidate Detection — {len(tagged_spans)} raw spans (pre mock-resolution)",
            "Not every candidate here survives to a redaction — see stage 6 for what got dropped and why",
        ),
        out_dir, 5, "raw PII candidates",
    )

    # ---- Stages 6-8: resolved + padded + brand-zone redactions -----------
    all_redactions, page_audits, ledger_rows, brand_zone_dicts = pipeline._collect_redactions(
        [state], opts, region_start=0
    )
    non_brand = [r for r in all_redactions if r.assignment_source != "brand"]
    brand = [r for r in all_redactions if r.assignment_source == "brand"]

    tight_img = canonical.original_image.convert("RGB").copy()
    draw = ImageDraw.Draw(tight_img)
    for r in non_brand:
        color = _entity_color(r.entity_type)
        b = r.canonical_bbox
        draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=color, width=3)
        _label_tag(draw, b.x, b.y, f"{r.entity_type} -> {r.mock_value}", color)
    dropped = len(tagged_spans) - len(non_brand)
    _draw_legend(
        tight_img,
        [(f"{et} ({sum(1 for r in non_brand if r.entity_type == et)})", _entity_color(et))
         for et in sorted({r.entity_type for r in non_brand})] or [("no redactions resolved", (200, 30, 30))],
    )
    _save(
        _with_title(
            tight_img,
            f"6. Mock Resolution + Dedup — {len(non_brand)} accepted redactions (tight box, pre-padding)",
            f"~{max(dropped, 0)} of {len(tagged_spans)} raw candidates did not survive resolution/dedup "
            f"(restrict_to_known_mappings={settings.restrict_to_known_mappings})",
        ),
        out_dir, 6, "accepted redactions (tight box)",
    )

    pad_img = canonical.original_image.convert("RGB").copy()
    draw = ImageDraw.Draw(pad_img)
    for r in non_brand:
        tb, pb = r.original_bbox, r.padded_bbox
        draw.rectangle([tb.x, tb.y, tb.x + tb.w, tb.y + tb.h], outline=(255, 150, 0), width=2)
        draw.rectangle([pb.x, pb.y, pb.x + pb.w, pb.y + pb.h], outline=(220, 30, 30), width=3)
    _draw_legend(
        pad_img,
        [("original/tight box", (255, 150, 0)), ("padded box (actually painted)", (220, 30, 30))],
    )
    _save(
        _with_title(
            pad_img,
            f"7. Padding — tight vs. padded box for each of {len(non_brand)} redactions",
            "A padded box that swallows a neighboring cell/word is an over-redaction gap; one that's still "
            "tight against the text is an under-padding gap",
        ),
        out_dir, 7, "padding comparison",
    )

    brand_img = canonical.original_image.convert("RGB").copy()
    draw = ImageDraw.Draw(brand_img)
    for r in non_brand:
        b = r.padded_bbox
        draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=(180, 180, 180), width=1)
    for r in brand:
        b = r.padded_bbox
        draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=(255, 200, 0), width=4)
        _label_tag(draw, b.x, b.y, r.entity_type, (200, 150, 0))
    _draw_legend(
        brand_img,
        [(f"brand/footer/image zone ({len(brand)})", (255, 200, 0)), ("PII redaction (context)", (180, 180, 180))],
    )
    _save(
        _with_title(
            brand_img,
            f"8. Brand Zone Detection — {len(brand)} footer/picture zones added",
            f"patch_footer={opts.patch_footer} patch_images={opts.patch_images}",
        ),
        out_dir, 8, "brand zones",
    )

    # ---- Stage 9: final rendered output -----------------------------------
    # Mirrors app/services/redact/pdf_renderer.py's own branch exactly: a
    # "digital" page with a real fitz_page is redacted as real vector PDF
    # content (pdf_native_redactor) and never rasterized; everything else
    # still gets the paint-then-flatten raster treatment. Rendering the
    # *actual* production path here (not a simulation of it) is what makes
    # this stage trustworthy for spotting real gaps.
    if page_kind == "digital" and raw_page.fitz_page is not None:
        from app.services.redact.pdf_native_redactor import (
            redact_image_regions,
            redact_text_regions,
        )

        clone_doc = fitz.open()
        clone_doc.insert_pdf(
            raw_page.fitz_page.parent,
            from_page=raw_page.fitz_page.number,
            to_page=raw_page.fitz_page.number,
        )
        clone_page = clone_doc[0]
        redact_text_regions(clone_page, all_redactions, opts.dpi)
        redact_image_regions(clone_page, all_redactions, opts.dpi)
        pix = clone_page.get_pixmap(dpi=opts.dpi, colorspace=fitz.csRGB)
        final_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        clone_doc.close()
        final_subtitle = (
            "page_kind=digital: REAL vector-native redaction (pdf_native_redactor) — "
            f"total redactions={len(all_redactions)} ({len(non_brand)} PII + {len(brand)} brand/footer/image) "
            "— output PDF page still has selectable text/real vector content, not a flattened image"
        )
    else:
        final_img = _draw_on_image(canonical.original_image, all_redactions)
        final_subtitle = (
            f"page_kind={page_kind}: paint-then-flatten raster path — "
            f"total redactions={len(all_redactions)} ({len(non_brand)} PII + {len(brand)} brand/footer/image)"
        )
    _save(
        _with_title(final_img, "9. Final Output — actual redaction path used in production", final_subtitle),
        out_dir, 9, "final redacted page",
    )

    print("\n=== Summary ===")
    print(f"page_kind={page_kind} blur_tier={t.blur_tier} structure_blocks={len(blocks)} ocr_words={len(ensemble_words)}")
    print(f"raw_candidates={len(tagged_spans)} (field={source_counts.get('field', 0)} "
          f"dictionary={source_counts.get('dictionary', 0)} "
          f"dictionary_fuzzy={source_counts.get('dictionary_fuzzy', 0)} "
          f"custom={source_counts.get('custom', 0)})")
    print(f"accepted_redactions={len(non_brand)} brand_zones={len(brand)}")
    print(f"\nImages written to: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
