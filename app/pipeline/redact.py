import logging
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import NamedTuple

from app.config import Settings, get_settings
from app.models.mock import (
    LedgerEntry,
    LedgerStoreProtocol,
    MockDictionaryStoreProtocol,
    MockValidationError,
    SubstitutionLedger,
)
from app.models.pii_chunk import BBox
from app.models.redact import (
    ConfidenceBreakdown,
    CustomRedactTerm,
    PageAuditSummary,
    RedactAuditResponse,
    RedactOptions,
    RedactionRegion,
)
from app.pipeline.errors import PipelineStageError
from app.pipeline.page_state import PageProcessState
from app.services.locale.resolver import resolve_languages
from app.services.ocr.ensemble import ensemble_ocr_page
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.ocr.page_renderer import load_pages
from app.services.pii.brand_zones import BrandZone, detect_brand_zones
from app.services.pii.coordinate_map import apply_padding, canonical_to_original
from app.services.pii.custom_redact import find_term_spans
from app.services.pii.detector import detect_pii
from app.services.pii.ensemble_mapper import (
    map_span_to_ensemble_bbox,
    union_bbox,
    words_for_span,
)
from app.services.pii.field_extractor import extract_field_candidates
from app.services.pii.mock_dictionary import normalize_source
from app.services.pii.redaction_scorer import score_redaction
from app.services.preprocess.canonical import CanonicalPage, canonicalize_page
from app.services.redact.audit_store import AuditStore
from app.services.redact.pdf_renderer import render_redacted_pdf
from app.services.redact.session_store import RedactSession, session_store
from app.services.structure.docling_adapter import DocBlock, extract_structure
from app.services.structure.spatial_join import join_words_to_blocks
from app.services.structure.table_geometry import extract_table_geometry, merge_table_geometry

logger = logging.getLogger(__name__)


class _SpanCandidate(NamedTuple):
    """A detected span awaiting mock-dictionary resolution.

    ``field_role``/``account_number`` are set for field-anchored candidates
    (see ``field_extractor``) and left ``None`` for custom terms and the
    optional legacy Presidio path.

    ``words`` carries the exact matched ``EnsembleWord``\\s for
    field-anchored candidates — geometry and source text are built
    directly from them rather than from ``start``/``end`` (a table cell's
    words are visually adjacent but commonly non-contiguous in
    ``merged_text``, so re-deriving words via character-overlap on that
    range would sweep in unrelated ones). Empty for custom terms and
    Presidio spans, which use the char-span path instead since their
    ranges are genuine contiguous slices of ``merged_text``.
    """

    start: int
    end: int
    entity_type: str
    score: float
    term: CustomRedactTerm | None
    field_role: str | None
    account_number: str | None
    words: tuple[EnsembleWord, ...] = ()


def _user_mock_for_term(term: CustomRedactTerm) -> str | None:
    """None if mock_label is blank or 'CUSTOM' (case-insensitive)."""
    label = (term.mock_label or "").strip()
    if not label or label.casefold() == "custom":
        return None
    return label


_PADDING_CELL_OVERLAP_THRESHOLD = 0.6


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    return bbox.x + bbox.w / 2, bbox.y + bbox.h / 2


def _point_in_bbox(bbox: BBox, x: float, y: float) -> bool:
    return bbox.x <= x <= bbox.x + bbox.w and bbox.y <= y <= bbox.y + bbox.h


def _bbox_overlap_fraction(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner``'s own area covered by ``outer``."""
    ix1, iy1 = max(inner.x, outer.x), max(inner.y, outer.y)
    ix2 = min(inner.x + inner.w, outer.x + outer.w)
    iy2 = min(inner.y + inner.h, outer.y + outer.h)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = inner.w * inner.h
    return inter / area if area > 0 else 0.0


def _enclosing_cell_bbox(
    bbox: BBox,
    blocks: list[DocBlock],
    word_bboxes: list[BBox] | None = None,
) -> BBox | None:
    """Table-cell bbox (canonical space) that owns this redaction.

    Prefers a majority vote of ``word_bboxes`` centers (each word picks
    the smallest cell containing its center) so an OCR/union box that
    spilled into a neighboring column still resolves to the cell the
    *name words* actually sit in — not the oversized union, and not
    "no cell" just because the union's overlap with any one cell dropped
    below ``_PADDING_CELL_OVERLAP_THRESHOLD``.

    Without word boxes: keep the original substantial-overlap test, then
    fall back to the smallest cell containing the union's center (the
    oversized-union case where no single cell covers 60% of the spill).
    """
    cells = [block for block in blocks if block.block_type == "cell"]
    if not cells:
        return None

    if word_bboxes:
        votes: dict[int, int] = {}
        bbox_by_id: dict[int, BBox] = {}
        for word_box in word_bboxes:
            if word_box.w <= 0 or word_box.h <= 0:
                continue
            cx, cy = _bbox_center(word_box)
            containing = [c for c in cells if _point_in_bbox(c.bbox, cx, cy)]
            if not containing:
                continue
            chosen = min(containing, key=lambda c: c.bbox.w * c.bbox.h)
            votes[id(chosen)] = votes.get(id(chosen), 0) + 1
            bbox_by_id[id(chosen)] = chosen.bbox
        if votes:
            winner = max(votes, key=votes.get)
            return bbox_by_id[winner]

    best_bbox: BBox | None = None
    best_frac = 0.0
    for cell in cells:
        frac = _bbox_overlap_fraction(bbox, cell.bbox)
        if frac > best_frac:
            best_frac, best_bbox = frac, cell.bbox
    if best_frac >= _PADDING_CELL_OVERLAP_THRESHOLD:
        return best_bbox

    cx, cy = _bbox_center(bbox)
    containing = [c for c in cells if _point_in_bbox(c.bbox, cx, cy)]
    if containing:
        return min(containing, key=lambda c: c.bbox.w * c.bbox.h).bbox
    return None


def _row_neighbor_clamp_x(
    bbox: BBox, words: list[EnsembleWord], excluded_ids: set[int]
) -> tuple[float | None, float | None]:
    """Midpoint x (canonical space) toward the nearest non-redacted word
    on the same visual row, on each side of ``bbox``.

    Used as the ``apply_padding`` clamp when no enclosing table cell is
    known (prose, labels, signature blocks) — the same midpoint-clamp
    idea ``field_extractor._split_multiword_tokens`` already uses for its
    interior sub-token boundaries, just measured against the nearest
    *real* neighboring word instead of an adjacent sub-token estimate.
    ``excluded_ids`` (the candidate's own matched words) are never
    counted as their own neighbor.
    """
    left_edge: float | None = None
    right_edge: float | None = None
    for w in words:
        if id(w) in excluded_ids or w.bbox.w <= 0:
            continue
        y_overlap = min(bbox.y + bbox.h, w.bbox.y + w.bbox.h) - max(bbox.y, w.bbox.y)
        if y_overlap <= 0.5 * min(bbox.h, w.bbox.h):
            continue  # not on the same visual row
        if w.bbox.x + w.bbox.w <= bbox.x:
            edge = w.bbox.x + w.bbox.w
            left_edge = edge if left_edge is None else max(left_edge, edge)
        elif w.bbox.x >= bbox.x + bbox.w:
            edge = w.bbox.x
            right_edge = edge if right_edge is None else min(right_edge, edge)
    left_x = (left_edge + bbox.x) / 2 if left_edge is not None else None
    right_x = (bbox.x + bbox.w + right_edge) / 2 if right_edge is not None else None
    return left_x, right_x


_LOOKAHEAD_MAX_EXTRA_WORDS = 3
_LOOKAHEAD_DECISIVE_MARGIN = 0.03


def _nearest_extension_word(
    bbox: BBox, words: list[EnsembleWord], excluded_ids: set[int]
) -> EnsembleWord | None:
    """The one word that most plausibly continues a value right after
    ``bbox``: the closest word to its right on the same visual row, or —
    when there is none — the closest word directly below it that's still
    roughly the same horizontal position.

    These are the two shapes a dropped continuation word actually takes
    in this pipeline's OCR-geometry output: the next word in an ordinary
    row, or the next line of a multi-line-wrapped table cell (see
    ``field_extractor._column_band_stitch``, which handles the *row-
    stitching* half of this same failure mode — this covers the
    *matching* half, for when a continuation word still ends up
    unattached to any candidate's word list).
    """
    same_row_right: tuple[float, EnsembleWord] | None = None
    below: tuple[float, EnsembleWord] | None = None
    for w in words:
        if id(w) in excluded_ids or w.bbox.w <= 0:
            continue
        y_overlap = min(bbox.y + bbox.h, w.bbox.y + w.bbox.h) - max(bbox.y, w.bbox.y)
        same_row = y_overlap > 0.5 * min(bbox.h, w.bbox.h)
        if same_row and w.bbox.x >= bbox.x + bbox.w:
            gap = w.bbox.x - (bbox.x + bbox.w)
            if same_row_right is None or gap < same_row_right[0]:
                same_row_right = (gap, w)
        elif not same_row and w.bbox.y >= bbox.y + bbox.h:
            x_overlap = min(bbox.x + bbox.w, w.bbox.x + w.bbox.w) - max(bbox.x, w.bbox.x)
            same_column_ish = x_overlap > 0 or abs(w.bbox.x - bbox.x) <= bbox.w
            if not same_column_ish:
                continue
            gap = w.bbox.y - (bbox.y + bbox.h)
            if below is None or gap < below[0]:
                below = (gap, w)
    if same_row_right is not None:
        return same_row_right[1]
    if below is not None:
        return below[1]
    return None


def _build_window_candidates(
    matched_words: list[EnsembleWord],
    ensemble_words: list[EnsembleWord],
    max_extra: int = _LOOKAHEAD_MAX_EXTRA_WORDS,
) -> list[list[EnsembleWord]]:
    """Candidate word windows for maximal-munch resolution: the original
    match, up to ``max_extra`` geometrically-extended versions (one more
    spatially-adjacent word each), and — when there's more than one word
    to drop — the original trimmed of its trailing word.

    This is a small local search over *segmentation boundaries*, the
    same idea as maximal-munch tokenization: none of these windows are
    assumed correct up front, ``_resolve_maximal_munch_window`` scores
    each one and picks the best.
    """
    if not matched_words:
        return []
    windows = [list(matched_words)]
    excluded_ids = {id(w) for w in matched_words}
    extended = list(matched_words)
    for _ in range(max_extra):
        nxt = _nearest_extension_word(union_bbox(extended), ensemble_words, excluded_ids)
        if nxt is None:
            break
        extended = extended + [nxt]
        excluded_ids.add(id(nxt))
        windows.append(list(extended))
    if len(matched_words) > 1:
        windows.append(list(matched_words[:-1]))
    return windows


def _resolve_maximal_munch_window(
    matched_words: list[EnsembleWord],
    ensemble_words: list[EnsembleWord],
    mock_store: MockDictionaryStoreProtocol,
    *,
    margin: float = _LOOKAHEAD_DECISIVE_MARGIN,
) -> tuple[list[EnsembleWord], str]:
    """Pick the best-scoring candidate word window for an ambiguous
    prefix-collision candidate (see ``MockDictionaryStoreProtocol
    .find_prefix_collisions``, the caller's ambiguity gate).

    Scores every window from ``_build_window_candidates`` against known
    dictionary entries via ``best_match_score`` (a pure read, so scoring
    a window that doesn't end up chosen never records a phantom hit), then
    picks the single best-scoring window — except when a *longer* window
    scores within ``margin`` of it, in which case the longer one wins.
    This directly implements the plan's "bias toward the longer window
    unless the shorter one scores decisively better": the longer window
    is preferred by default, and only a decisive (>margin) score
    advantage for a shorter one overrides that bias.

    Returns:
        ``(words, joined_text)`` for the chosen window. Falls back to
        ``matched_words`` unchanged (as ``(matched_words, joined_text)``)
        when there's nothing to search (``matched_words`` empty/singleton
        with no extension available).
    """
    windows = _build_window_candidates(matched_words, ensemble_words)
    if not windows:
        return matched_words, " ".join(w.text for w in matched_words)

    scored: list[tuple[list[EnsembleWord], str, float]] = []
    for window in windows:
        text = " ".join(w.text for w in window)
        normalized = normalize_source(text)
        ratio = mock_store.best_match_score(normalized) if normalized else 0.0
        scored.append((window, text, ratio))

    best_ratio = max(ratio for _, _, ratio in scored)
    within_margin = [item for item in scored if best_ratio - item[2] <= margin]
    chosen_words, chosen_text, _ = max(within_margin, key=lambda item: len(item[0]))
    return chosen_words, chosen_text


_NAME_SHAPED_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
_SPILLOVER_ROW_OVERLAP_FRACTION = 0.5
_SPILLOVER_MAX_GAP_FACTOR = 1.5
_SPILLOVER_COVERAGE_THRESHOLD = 0.5


def _is_name_shaped(word: EnsembleWord) -> bool:
    """True for an alphabetic, capitalized word — the shape a dropped
    name/org-name continuation actually takes (e.g. a stray "Plus"), as
    opposed to punctuation, bare digits, or a lowercase function word
    that's unlikely to itself be PII worth a fail-safe absorb.
    """
    text = word.text.strip()
    return bool(_NAME_SHAPED_RE.fullmatch(text)) and text[0].isupper()


def _union_bboxes(a: BBox, b: BBox) -> BBox:
    x1, y1 = min(a.x, b.x), min(a.y, b.y)
    x2 = max(a.x + a.w, b.x + b.w)
    y2 = max(a.y + a.h, b.y + b.h)
    return BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _nearest_adjacent_redaction(
    bbox: BBox, redactions: list[RedactionRegion]
) -> RedactionRegion | None:
    """The already-redacted region immediately to the left or right of
    ``bbox`` on the same visual row, within a small gap — the "sits
    immediately next to an already-matched-and-redacted span" test from
    the plan's spillover fail-safe. ``None`` if nothing qualifies.
    """
    best: tuple[float, RedactionRegion] | None = None
    for region in redactions:
        rb = region.canonical_bbox
        y_overlap = min(bbox.y + bbox.h, rb.y + rb.h) - max(bbox.y, rb.y)
        if y_overlap <= _SPILLOVER_ROW_OVERLAP_FRACTION * min(bbox.h, rb.h):
            continue
        if bbox.x >= rb.x + rb.w:
            gap = bbox.x - (rb.x + rb.w)
        elif rb.x >= bbox.x + bbox.w:
            gap = rb.x - (bbox.x + bbox.w)
        else:
            continue
        if gap > _SPILLOVER_MAX_GAP_FACTOR * max(bbox.h, 1):
            continue
        if best is None or gap < best[0]:
            best = (gap, region)
    return best[1] if best else None


def _apply_spillover_safety_net(
    page_redactions: list[RedactionRegion],
    ensemble_words: list[EnsembleWord],
    blocks: list[DocBlock],
    canonical: CanonicalPage,
) -> None:
    """Absorb an orphaned name-shaped word into an adjacent redaction's
    bounding box, per the plan's spillover fail-safe: geometry fixes and
    the maximal-munch probe reduce how often OCR/stitching drops a word
    entirely, but can't eliminate it, so this is the last-resort net —
    it never creates a new redaction or dictionary entry, only extends an
    existing region's bounds (canonical/original/padded bbox) to also
    cover the orphan. Mutates ``page_redactions`` in place.
    """
    for word in ensemble_words:
        if word.bbox.w <= 0 or word.bbox.h <= 0 or not _is_name_shaped(word):
            continue
        if any(
            _bbox_overlap_fraction(word.bbox, r.canonical_bbox) >= _SPILLOVER_COVERAGE_THRESHOLD
            for r in page_redactions
        ):
            continue
        target = _nearest_adjacent_redaction(word.bbox, page_redactions)
        if target is None:
            continue

        # Don't pull a word from a neighboring *cell* into this redaction —
        # that is how a name patch grows across a gridline and covers the
        # USD/amount value next door. Same-cell continuations (wrapped
        # fund names) are still absorbed.
        target_cell = _enclosing_cell_bbox(
            target.canonical_bbox,
            blocks,
            word_bboxes=[target.canonical_bbox],
        )
        if target_cell is not None:
            wx, wy = _bbox_center(word.bbox)
            if not _point_in_bbox(target_cell, wx, wy):
                continue

        extended_bbox = _union_bboxes(target.canonical_bbox, word.bbox)
        cell_bbox = target_cell or _enclosing_cell_bbox(
            extended_bbox, blocks, word_bboxes=[target.canonical_bbox, word.bbox]
        )
        left_neighbor_x = right_neighbor_x = None
        original_cell_bbox = None
        if cell_bbox is not None:
            original_cell_bbox = canonical_to_original(cell_bbox, canonical.transform)
        else:
            excluded_ids = {
                id(w)
                for w in ensemble_words
                if _bbox_overlap_fraction(w.bbox, extended_bbox) >= _SPILLOVER_COVERAGE_THRESHOLD
            }
            left_x, right_x = _row_neighbor_clamp_x(extended_bbox, ensemble_words, excluded_ids)
            dx = canonical.transform.dx
            left_neighbor_x = left_x + dx if left_x is not None else None
            right_neighbor_x = right_x + dx if right_x is not None else None

        target.canonical_bbox = extended_bbox
        target.original_bbox = canonical_to_original(extended_bbox, canonical.transform)
        target.padded_bbox = apply_padding(
            target.original_bbox,
            canonical.transform.blur_tier,
            canonical.original_image.width,
            canonical.original_image.height,
            cell_bbox=original_cell_bbox,
            left_neighbor_x=left_neighbor_x,
            right_neighbor_x=right_neighbor_x,
        )
        target.engines_seen = sorted(set(target.engines_seen) | set(word.engines))


def _brand_region(zone: BrandZone, blur_tier: str, region_id: str) -> RedactionRegion:
    """Map BrandZone → RedactionRegion. Dummy scores; no dictionary row."""
    entity_type = "BRAND_LOGO" if zone.zone == "logo" else "BRAND_FOOTER"
    dummy = ConfidenceBreakdown(
        presidio=0.0,
        ocr=0.0,
        engine_agreement=0.0,
        structural_context=0.0,
    )
    return RedactionRegion(
        region_id=region_id,
        page=zone.page,
        entity_type=entity_type,
        canonical_bbox=zone.bbox,
        original_bbox=zone.bbox,
        padded_bbox=zone.bbox,
        redaction_confidence=0.0,
        confidence_breakdown=dummy,
        structural_context=None,
        blur_tier=blur_tier,  # type: ignore[arg-type]
        engines_seen=[],
        mock_value=zone.label,
        mapping_id=None,
        assignment_source="brand",
    )


def _dedupe_ledger_entries(rows: list[dict]) -> list[LedgerEntry]:
    """Collapse scratch rows by mapping_id and union page lists."""
    by_id: dict[str, LedgerEntry] = {}
    for row in rows:
        mapping_id = row["mapping_id"]
        page = row["page"]
        existing = by_id.get(mapping_id)
        if existing is None:
            by_id[mapping_id] = LedgerEntry(
                mapping_id=mapping_id,
                source_text=row["source_text"],
                mock_value=row["mock_value"],
                entity_type=row["entity_type"],
                assignment_source=row["assignment_source"],
                hit_count=row["hit_count"],
                pages=[page],
            )
            continue
        if page not in existing.pages:
            existing.pages.append(page)
        existing.hit_count = row["hit_count"]
        existing.mock_value = row["mock_value"]
        existing.assignment_source = row["assignment_source"]
    return list(by_id.values())


class RedactPipeline:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        mock_store: MockDictionaryStoreProtocol | None = None,
        ledger_store: LedgerStoreProtocol | None = None,
        audit_store: AuditStore | None = None,
    ) -> None:
        from app.api.mock_routes import get_ledger_store, get_mock_store

        self.settings = settings or get_settings()
        self.mock_store = mock_store or get_mock_store()
        self.ledger_store = ledger_store or get_ledger_store()
        self.audit_store = audit_store or AuditStore()

    async def run(
        self,
        file_bytes: bytes,
        filename: str,
        options: RedactOptions | None = None,
    ) -> tuple[bytes, RedactAuditResponse, RedactSession]:
        opts = options or RedactOptions()
        started = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        session_id = uuid.uuid4().hex

        page_states = await self._build_page_states(file_bytes, filename, opts)
        all_redactions, page_audits, ledger_rows, brand_zones = self._collect_redactions(
            page_states, opts, region_start=0
        )

        if opts.strict_pii and not all_redactions:
            raise PipelineStageError("pii_join", "No PII detected (strict_pii=true)")

        pdf_bytes = self._render_pdf(page_states, all_redactions, filename)
        audit = self._build_audit(
            request_id, filename, page_states, page_audits, all_redactions, started
        )
        self._persist_outputs(audit, ledger_rows, brand_zones)

        session = RedactSession(
            session_id=session_id,
            file_bytes=file_bytes,
            filename=filename,
            options=opts,
            page_states=page_states,
            last_pdf=pdf_bytes,
            last_audit=audit,
            custom_terms=[t.search_value for t in opts.custom_redactions],
        )
        session_store.put(session)
        return pdf_bytes, audit, session

    async def regenerate(
        self,
        session_id: str,
        custom_terms_text: str | None = None,
        options: RedactOptions | None = None,
    ) -> tuple[bytes, RedactAuditResponse, RedactSession]:
        session = session_store.get(session_id)
        if not session or not session.page_states:
            raise PipelineStageError("session", "Session not found or expired — upload and process again")

        from app.services.pii.custom_redact import parse_custom_terms

        opts = options or session.options
        if custom_terms_text is not None:
            opts = opts.model_copy(update={"custom_redactions": parse_custom_terms(custom_terms_text)})

        started = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        all_redactions, page_audits, ledger_rows, brand_zones = self._collect_redactions(
            session.page_states, opts, region_start=0
        )
        pdf_bytes = self._render_pdf(session.page_states, all_redactions, session.filename)
        audit = self._build_audit(
            request_id, session.filename, session.page_states, page_audits, all_redactions, started
        )
        self._persist_outputs(audit, ledger_rows, brand_zones)

        session.options = opts
        session.last_pdf = pdf_bytes
        session.last_audit = audit
        session.custom_terms = [t.search_value for t in opts.custom_redactions]
        session_store.put(session)
        return pdf_bytes, audit, session

    def _persist_outputs(
        self,
        audit: RedactAuditResponse,
        ledger_rows: list[dict],
        brand_zones: list[dict],
    ) -> None:
        ledger = SubstitutionLedger(
            request_id=audit.request_id,
            created_at=audit.created_at,
            entries=_dedupe_ledger_entries(ledger_rows),
            brand_zones=brand_zones,
        )
        try:
            self.ledger_store.save(ledger)
        except Exception as exc:
            raise PipelineStageError("ledger", "Failed to save substitution ledger") from exc
        self.audit_store.save(audit)
        self.audit_store.log_summary(audit)

    async def _build_page_states(
        self,
        file_bytes: bytes,
        filename: str,
        opts: RedactOptions,
    ) -> list[PageProcessState]:
        try:
            pages = load_pages(file_bytes, filename, dpi=opts.dpi)
        except Exception as exc:
            raise PipelineStageError("ingest", "Failed to load document pages", exc) from exc

        if not pages:
            raise PipelineStageError("ingest", "Document has no pages")

        if len(pages) > self.settings.max_pages_per_document:
            raise PipelineStageError(
                "ingest",
                f"Document exceeds max pages ({self.settings.max_pages_per_document})",
            )

        canonical_pages: list[CanonicalPage] = []
        try:
            for idx, page in enumerate(pages):
                canonical_pages.append(
                    canonicalize_page(
                        page,
                        idx,
                        strip_gridlines=self.settings.strip_gridlines_enabled,
                        deskew=self.settings.deskew_enabled,
                    )
                )
        except Exception as exc:
            raise PipelineStageError("preprocess", "Preprocessing failed", exc) from exc

        sample_text = ""
        locale, langs, tess_lang = resolve_languages(opts.locale, opts.languages, opts.auto_detect, sample_text)
        page_states: list[PageProcessState] = []

        for canonical in canonical_pages:
            try:
                # Structure extraction (Docling + img2table below) runs on
                # original_image, not the line-stripped canonical_image OCR
                # uses: both rely on visible table borders to find cell
                # boundaries, and canonical_image's strip_table_lines erases
                # essentially every rule on tightly-ruled tables like these —
                # handing the structure model strictly less information than
                # the scan had. original_image is pixel-registered 1:1 with
                # canonical_image (dx=dy=0), so bboxes from either are
                # directly comparable with no translation needed.
                blocks = extract_structure(canonical.original_image)
            except Exception as exc:
                raise PipelineStageError("docling", "Structure extraction failed", exc) from exc

            if self.settings.img2table_enabled:
                try:
                    img2table_blocks = extract_table_geometry(canonical.original_image)
                    blocks = merge_table_geometry(
                        blocks,
                        img2table_blocks,
                        min_table_iou=self.settings.img2table_min_table_iou,
                    )
                except Exception:
                    logger.warning("img2table merge failed, keeping Docling-only blocks", exc_info=False)

            # Table/cell geometry from structure extraction (image-only, so it
            # can run before OCR) lets the ensemble merge favor a
            # table-friendlier engine when word readings disagree inside a
            # table — see ensemble.align_word_boxes's table_regions bias.
            table_regions = [b.bbox for b in blocks if b.block_type in ("table", "cell")]

            try:
                merged_text, ensemble_words, _ = await ensemble_ocr_page(
                    canonical.canonical_image,
                    canonical.page_index,
                    tess_lang,
                    langs,
                    table_regions=table_regions,
                    engine_filter=opts.ocr_engines,
                )
            except Exception as exc:
                raise PipelineStageError("ensemble_ocr", "OCR ensemble failed", exc) from exc

            sample_text += merged_text + "\n"
            if opts.auto_detect and canonical.page_index == 0 and not opts.locale:
                locale, langs, tess_lang = resolve_languages(None, None, True, sample_text)

            try:
                word_context = join_words_to_blocks(ensemble_words, blocks)
            except Exception as exc:
                raise PipelineStageError("docling", "Structure extraction failed", exc) from exc

            page_states.append(
                PageProcessState(
                    canonical=canonical,
                    merged_text=merged_text,
                    ensemble_words=ensemble_words,
                    blocks=blocks,
                    word_context=word_context,
                )
            )

        return page_states

    def _upsert_custom_overrides(self, opts: RedactOptions) -> None:
        for term in opts.custom_redactions:
            user_mock = _user_mock_for_term(term)
            if user_mock is None:
                continue
            try:
                self.mock_store.upsert(term.search_value, user_mock)
            except MockValidationError:
                continue

    def _collect_redactions(
        self,
        page_states: list[PageProcessState],
        opts: RedactOptions,
        region_start: int,
    ) -> tuple[list[RedactionRegion], list[PageAuditSummary], list[dict], list[dict]]:
        locale, _, _ = resolve_languages(opts.locale, opts.languages, opts.auto_detect, "")
        all_redactions: list[RedactionRegion] = []
        page_audits: list[PageAuditSummary] = []
        ledger_rows: list[dict] = []
        brand_zone_dicts: list[dict] = []
        region_counter = region_start
        seen_keys: set[tuple[int, int, int, int, str]] = set()

        self._upsert_custom_overrides(opts)

        for state in page_states:
            canonical = state.canonical
            merged_text = state.merged_text
            ensemble_words = state.ensemble_words
            word_context = state.word_context
            page_redactions: list[RedactionRegion] = []

            spans: list[_SpanCandidate] = []

            if self.settings.field_detection_enabled:
                stitch_blocks = state.blocks if self.settings.docling_cell_stitch_enabled else None
                for cand in extract_field_candidates(
                    merged_text, ensemble_words, stitch_blocks, word_context
                ):
                    spans.append(
                        _SpanCandidate(
                            cand.start,
                            cand.end,
                            cand.entity_type,
                            cand.match_confidence,
                            None,
                            cand.field_role,
                            cand.account_number,
                            cand.words,
                        )
                    )

            if self.settings.presidio_enabled:
                for d in detect_pii(merged_text, locale):
                    spans.append(
                        _SpanCandidate(d["start"], d["end"], d["entity_type"], d["score"], None, None, None)
                    )

            for term in opts.custom_redactions:
                for start, end in find_term_spans(merged_text, term.search_value):
                    spans.append(_SpanCandidate(start, end, "CUSTOM", 0.95, term, None, None))

            for start, end, entity_type, score, term, _field_role, _account_number, cand_words in spans:
                if cand_words:
                    # Field-anchored candidate: geometry/text come straight from
                    # the words the extractor matched, not from start/end — a
                    # table cell's words are non-contiguous in merged_text (see
                    # _SpanCandidate docstring), so re-deriving via char-overlap
                    # would silently sweep in unrelated neighboring words.
                    matched_words = list(cand_words)
                    bbox = union_bbox(matched_words)
                    source_text = " ".join(w.text for w in matched_words)
                else:
                    bbox = map_span_to_ensemble_bbox(start, end, ensemble_words, merged_text)
                    matched_words = words_for_span(start, end, ensemble_words) if bbox else []
                    source_text = merged_text[start:end]
                if bbox is None:
                    continue

                if matched_words:
                    # Maximal-munch resolution: a prefix-collision hit here
                    # (e.g. "Maksima" vs. "Maksima Plus") means the plain
                    # match is ambiguous — probe geometrically-adjacent word
                    # windows and let the one that best matches a known
                    # entry win, biased toward the longer window (see
                    # _resolve_maximal_munch_window).
                    normalized_candidate = normalize_source(source_text)
                    if normalized_candidate and self.mock_store.find_prefix_collisions(
                        normalized_candidate
                    ):
                        matched_words, source_text = _resolve_maximal_munch_window(
                            matched_words, ensemble_words, self.mock_store
                        )
                        bbox = union_bbox(matched_words)

                original_bbox = canonical_to_original(bbox, canonical.transform)
                word_bboxes = [w.bbox for w in matched_words] if matched_words else None
                cell_bbox = _enclosing_cell_bbox(bbox, state.blocks, word_bboxes=word_bboxes)
                left_neighbor_x = right_neighbor_x = None
                original_cell_bbox = None
                if cell_bbox is not None:
                    original_cell_bbox = canonical_to_original(cell_bbox, canonical.transform)
                else:
                    excluded_ids = {id(w) for w in matched_words}
                    left_x, right_x = _row_neighbor_clamp_x(bbox, ensemble_words, excluded_ids)
                    dx = canonical.transform.dx
                    left_neighbor_x = left_x + dx if left_x is not None else None
                    right_neighbor_x = right_x + dx if right_x is not None else None
                padded = apply_padding(
                    original_bbox,
                    canonical.transform.blur_tier,
                    canonical.original_image.width,
                    canonical.original_image.height,
                    cell_bbox=original_cell_bbox,
                    left_neighbor_x=left_neighbor_x,
                    right_neighbor_x=right_neighbor_x,
                )
                key = (canonical.page_index, padded.x, padded.y, padded.w, entity_type)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                if not source_text.strip():
                    continue

                user_mock = _user_mock_for_term(term) if term is not None else None
                if term is None and self.settings.restrict_to_known_mappings:
                    # Field-anchored/Presidio candidate under restricted mode:
                    # only paint it if it already matches the curated
                    # dictionary (exact or trusted fuzzy) — never
                    # auto-create a new entry for unseen text. Explicit
                    # custom redaction terms (term is not None) are a direct
                    # user instruction and always resolve/create regardless.
                    entry = self.mock_store.lookup(source_text)
                    if entry is None:
                        continue
                else:
                    entry = self.mock_store.resolve(source_text, user_mock=user_mock)
                logger.info(
                    "mock_resolve mapping_id=%s entity_type=%s assignment_source=%s",
                    entry.mapping_id,
                    entity_type,
                    entry.assignment_source,
                )

                ctx = None
                if matched_words:
                    # Look up by char-offset containment, not list membership/
                    # equality: field-anchored candidates match sub-words split
                    # out of an original ensemble word (see field_extractor's
                    # _split_multiword_tokens), which are equal to no entry in
                    # ensemble_words itself.
                    first_char = matched_words[0].char_start
                    first_idx = next(
                        (
                            idx
                            for idx, w in enumerate(ensemble_words)
                            if w.char_start <= first_char < w.char_end
                        ),
                        None,
                    )
                    if first_idx is not None:
                        ctx = word_context.get(first_idx)

                confidence, breakdown = score_redaction(score, matched_words, ctx)
                region_counter += 1
                page_redactions.append(
                    RedactionRegion(
                        region_id=f"r-{region_counter:04d}",
                        page=canonical.page_index,
                        entity_type=entity_type,
                        canonical_bbox=bbox,
                        original_bbox=original_bbox,
                        padded_bbox=padded,
                        redaction_confidence=confidence,
                        confidence_breakdown=breakdown,
                        structural_context=ctx,
                        blur_tier=canonical.transform.blur_tier,
                        engines_seen=sorted({e for w in matched_words for e in w.engines}),
                        mock_value=entry.mock_value,
                        mapping_id=entry.mapping_id,
                        assignment_source=entry.assignment_source,
                    )
                )
                ledger_rows.append(
                    {
                        "mapping_id": entry.mapping_id,
                        "source_text": source_text,
                        "mock_value": entry.mock_value,
                        "entity_type": entity_type,
                        "assignment_source": entry.assignment_source,
                        "hit_count": entry.hit_count,
                        "page": canonical.page_index,
                    }
                )

            if self.settings.spillover_safety_net_enabled and page_redactions:
                _apply_spillover_safety_net(page_redactions, ensemble_words, state.blocks, canonical)

            original = canonical.original_image
            zones = detect_brand_zones(
                page_w=original.width,
                page_h=original.height,
                page=canonical.page_index,
                blocks=state.blocks,
                patch_logo=opts.patch_logo,
                patch_footer=opts.patch_footer,
                logo_top_pct=self.settings.logo_zone_top_pct,
                logo_right_pct=self.settings.logo_zone_right_pct,
                footer_bottom_pct=self.settings.footer_zone_bottom_pct,
            )
            for zone in zones:
                region_counter += 1
                page_redactions.append(
                    _brand_region(zone, canonical.transform.blur_tier, f"r-{region_counter:04d}")
                )
                brand_zone_dicts.append(
                    {
                        "page": zone.page,
                        "zone": zone.zone,
                        "bbox": zone.bbox.model_dump(),
                    }
                )

            all_redactions.extend(page_redactions)
            page_audits.append(
                PageAuditSummary(
                    page=canonical.page_index,
                    blur_tier=canonical.transform.blur_tier,
                    blur_variance=canonical.transform.blur_variance,
                    transform=canonical.transform,
                    ensemble_word_count=len(ensemble_words),
                    docling_block_count=len(state.blocks),
                    redaction_count=len(page_redactions),
                )
            )

        return all_redactions, page_audits, ledger_rows, brand_zone_dicts

    def _render_pdf(
        self,
        page_states: list[PageProcessState],
        redactions: list[RedactionRegion],
        filename: str,
    ) -> bytes:
        try:
            return render_redacted_pdf(
                [s.canonical.original_image for s in page_states],
                redactions,
                filename,
            )
        except Exception as exc:
            raise PipelineStageError("redact_render", "Failed to render redacted PDF", exc) from exc

    def _build_audit(
        self,
        request_id: str,
        filename: str,
        page_states: list[PageProcessState],
        page_audits: list[PageAuditSummary],
        all_redactions: list[RedactionRegion],
        started: float,
    ) -> RedactAuditResponse:
        processing_ms = int((time.perf_counter() - started) * 1000)
        blur_tiers = Counter(p.blur_tier for p in page_audits)
        avg_conf = (
            sum(r.redaction_confidence for r in all_redactions) / len(all_redactions)
            if all_redactions
            else 0.0
        )
        return RedactAuditResponse(
            request_id=request_id,
            filename=filename,
            page_count=len(page_states),
            processing_ms=processing_ms,
            created_at=datetime.now(timezone.utc),
            summary={
                "redaction_count": len(all_redactions),
                "avg_confidence": round(avg_conf, 4),
                "blur_tiers": dict(blur_tiers),
            },
            pages=page_audits,
            redactions=all_redactions,
        )
