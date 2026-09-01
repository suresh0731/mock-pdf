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
from app.services.ocr.native_text import classify_and_extract, extract_native_words
from app.services.ocr.page_renderer import load_pages
from app.services.pii.brand_zones import (
    BrandZone,
    detect_brand_zones,
    detect_picture_zones,
    reconcile_picture_zones_with_text,
)
from app.services.pii.coordinate_map import apply_padding, canonical_to_original
from app.services.pii.custom_redact import find_fuzzy_term_spans, find_term_spans
from app.services.pii.ensemble_mapper import (
    map_span_to_ensemble_bbox,
    union_bbox,
    words_for_span,
)
from app.services.pii.field_extractor import extract_field_candidates
from app.services.pii.mock_dictionary import normalize_source
from app.services.pii.name_matcher import token_sort_ratio
from app.services.pii.redaction_scorer import score_redaction
from app.services.pii.signature_zones import detect_signature_zones
from app.services.preprocess.canonical import CanonicalPage, canonicalize_page
from app.services.redact.audit_store import AuditStore
from app.services.redact.ocr_output_store import OcrOutputStore
from app.services.redact.pdf_renderer import PageRenderInput, render_redacted_pdf
from app.services.redact.session_store import RedactSession, session_store
from app.services.structure.docling_adapter import DocBlock, extract_structure
from app.services.structure.spatial_join import join_words_to_blocks
from app.services.structure.table_geometry import extract_table_geometry, merge_table_geometry

logger = logging.getLogger(__name__)


class _SpanCandidate(NamedTuple):
    """A detected span awaiting mock-dictionary resolution.

    ``field_role``/``account_number`` are set for field-anchored candidates
    (see ``field_extractor``) and left ``None`` for custom terms and
    dictionary-scan hits (see ``dictionary_scan_enabled``).

    ``words`` carries the exact matched ``EnsembleWord``\\s for
    field-anchored candidates — geometry and source text are built
    directly from them rather than from ``start``/``end`` (a table cell's
    words are visually adjacent but commonly non-contiguous in
    ``merged_text``, so re-deriving words via character-overlap on that
    range would sweep in unrelated ones). Empty for custom terms and
    dictionary-scan spans, which use the char-span path instead since their
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


def _filter_words_to_cell(
    matched_words: list[EnsembleWord], cell_bbox: BBox
) -> list[EnsembleWord] | None:
    """Drop any word whose own center falls outside ``cell_bbox``.

    A genuine multi-line wrap keeps every one of its words inside the
    same cell; a word that doesn't belong (see the call site's comment
    on the reading-order/merged-text corruption this guards against) is
    dropped rather than trusted. Returns ``None`` when every word is
    already coherent (nothing changed, caller keeps its existing bbox)
    or when filtering would remove every word (an all-or-nothing miss
    means the cell itself is unreliable for this match, not that one
    stray word snuck in — safer to leave the original union untouched
    than to discard the whole match's geometry).

    Args:
        matched_words: Words backing the current union bbox.
        cell_bbox: The table cell the union bbox resolved to.

    Returns:
        The coherent-only subset, or ``None`` if no filtering should be
        applied.
    """
    if not matched_words:
        return None
    coherent = [w for w in matched_words if _point_in_bbox(cell_bbox, *_bbox_center(w.bbox))]
    if not coherent or len(coherent) == len(matched_words):
        return None
    return coherent


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
# Same scale as the spillover fail-safe: one line-height (plus a little
# slack) is a plausible gap between words of the *same* value; anything
# wider is the next table column or the next row, which must not be
# absorbed into this redaction.
_EXTENSION_MAX_GAP_FACTOR = 1.5
# A later detector (dictionary-scan of a nested short alias, an
# over-extended maximal-munch window) that substantially overlaps an
# already-painted box is skipped rather than stacking a second patch.
_PADDED_OVERLAP_SKIP_THRESHOLD = 0.5


def _nearest_extension_word(
    bbox: BBox,
    words: list[EnsembleWord],
    excluded_ids: set[int],
    *,
    origin_cell: BBox | None = None,
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

    A candidate is rejected when it sits farther than
    ``_EXTENSION_MAX_GAP_FACTOR`` line-heights away (the next column /
    next row, not a continuation) or, when ``origin_cell`` is known,
    when its center falls outside that cell — so a table-cell name is
    never extended into the neighboring amount/account-number column.
    """
    same_row_right: tuple[float, EnsembleWord] | None = None
    below: tuple[float, EnsembleWord] | None = None
    max_gap = _EXTENSION_MAX_GAP_FACTOR * max(bbox.h, 1)
    for w in words:
        if id(w) in excluded_ids or w.bbox.w <= 0:
            continue
        if origin_cell is not None:
            cx, cy = _bbox_center(w.bbox)
            if not _point_in_bbox(origin_cell, cx, cy):
                continue
        y_overlap = min(bbox.y + bbox.h, w.bbox.y + w.bbox.h) - max(bbox.y, w.bbox.y)
        same_row = y_overlap > 0.5 * min(bbox.h, w.bbox.h)
        if same_row and w.bbox.x >= bbox.x + bbox.w:
            gap = w.bbox.x - (bbox.x + bbox.w)
            if gap > max_gap:
                continue
            if same_row_right is None or gap < same_row_right[0]:
                same_row_right = (gap, w)
        elif not same_row and w.bbox.y >= bbox.y + bbox.h:
            x_overlap = min(bbox.x + bbox.w, w.bbox.x + w.bbox.w) - max(bbox.x, w.bbox.x)
            same_column_ish = x_overlap > 0 or abs(w.bbox.x - bbox.x) <= bbox.w
            if not same_column_ish:
                continue
            gap = w.bbox.y - (bbox.y + bbox.h)
            if gap > max_gap:
                continue
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
    *,
    origin_cell: BBox | None = None,
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
        nxt = _nearest_extension_word(
            union_bbox(extended), ensemble_words, excluded_ids, origin_cell=origin_cell
        )
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
    blocks: list[DocBlock] | None = None,
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

    When ``blocks`` contains table cells, extension words are restricted
    to the cell that owns the original match so a prefix like
    ``"Standard Chartered"`` cannot swallow the next column.

    Returns:
        ``(words, joined_text)`` for the chosen window. Falls back to
        ``matched_words`` unchanged (as ``(matched_words, joined_text)``)
        when there's nothing to search (``matched_words`` empty/singleton
        with no extension available).
    """
    origin_cell = None
    if blocks and matched_words:
        seed = union_bbox(matched_words)
        if seed is not None:
            origin_cell = _enclosing_cell_bbox(
                seed, blocks, word_bboxes=[w.bbox for w in matched_words]
            )
    windows = _build_window_candidates(
        matched_words, ensemble_words, origin_cell=origin_cell
    )
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


def _char_ranges_overlap(start: int, end: int, claimed: list[tuple[int, int]]) -> bool:
    """True if ``[start, end)`` overlaps any already-claimed char range."""
    return any(start < claimed_end and end > claimed_start for claimed_start, claimed_end in claimed)


# Placeholder used to blank out already-claimed text before a fuzzy scan
# (see _mask_claimed_ranges). Never alphanumeric, so it can't itself read
# as a plausible match for anything; repeating it (rather than any other
# filler) keeps the masked text the same length so char offsets found in
# it still index correctly into the original merged_text.
_FUZZY_MASK_CHAR = "\x00"


def _mask_claimed_ranges(text: str, claimed: list[tuple[int, int]]) -> str:
    """Blank out ``claimed`` char ranges in ``text`` with a placeholder.

    A dictionary entry that appears on a page both cleanly (already
    exact-matched elsewhere, e.g. several clean table rows) and garbled
    in one or more other spots (the case this masking exists for) must
    still have those garbled occurrences found by the fuzzy pass — but
    ``find_fuzzy_term_spans`` only probes a bounded number of occurrences
    per call, and a clean/near-exact occurrence always outscores a
    garbled one, so it would win every one of those slots before a
    different, still-uncaught garbled one ever got a turn. Masking every
    already-claimed range (from exact matches, custom terms, and any
    fuzzy match already accepted) before each fuzzy probe removes the
    clean occurrences from contention entirely, so the whole per-entry
    probe budget goes to text nothing has claimed yet.

    Args:
        text: Original merged page text (never logged).
        claimed: Already-claimed ``(start, end)`` char ranges.

    Returns:
        Same length as ``text``, with every claimed range replaced by
        ``_FUZZY_MASK_CHAR``. Unclaimed text is returned unchanged.
    """
    if not claimed:
        return text
    chars = list(text)
    for start, end in claimed:
        for i in range(max(start, 0), min(end, len(chars))):
            chars[i] = _FUZZY_MASK_CHAR
    return "".join(chars)


def _redaction_boxes_conflict(
    a: BBox, b: BBox, threshold: float = _PADDED_OVERLAP_SKIP_THRESHOLD
) -> bool:
    """True when either box is substantially covered by the other.

    Catches both near-duplicate patches (high IoU) and a later oversized
    box that fully contains an earlier, tighter one — the dictionary-scan
    + maximal-munch failure mode that stacked a second patch over an
    already-redacted table cell.
    """
    return (
        _bbox_overlap_fraction(a, b) >= threshold
        or _bbox_overlap_fraction(b, a) >= threshold
    )


def _padded_overlaps_existing(
    padded: BBox, existing: list[RedactionRegion]
) -> bool:
    return any(_redaction_boxes_conflict(padded, region.padded_bbox) for region in existing)


def _is_name_shaped(word: EnsembleWord, stopwords: frozenset[str] = frozenset()) -> bool:
    """True for an alphabetic, capitalized word — the shape a dropped
    name/org-name continuation actually takes (e.g. a stray "Plus"), as
    opposed to punctuation, bare digits, or a lowercase function word
    that's unlikely to itself be PII worth a fail-safe absorb.

    ``stopwords`` (case-insensitive, see ``Settings.spillover_non_name_
    stopwords``) excludes common document/tax abbreviations that are
    *also* this exact shape (capitalized, alphabetic-only — e.g. "DPP",
    "PPN") but are never themselves a name continuation, so they must
    never be swallowed into a neighboring redaction no matter how close
    they sit to one.
    """
    text = word.text.strip()
    if not (bool(_NAME_SHAPED_RE.fullmatch(text)) and text[0].isupper()):
        return False
    return text.upper() not in stopwords


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
    stopwords: frozenset[str] = frozenset(),
) -> None:
    """Absorb an orphaned name-shaped word into an adjacent redaction's
    bounding box, per the plan's spillover fail-safe: geometry fixes and
    the maximal-munch probe reduce how often OCR/stitching drops a word
    entirely, but can't eliminate it, so this is the last-resort net —
    it never creates a new redaction or dictionary entry, only extends an
    existing region's bounds (canonical/original/padded bbox) to also
    cover the orphan. Mutates ``page_redactions`` in place.

    ``stopwords`` (see ``_is_name_shaped``) keeps this net from also
    swallowing a common document/tax abbreviation (e.g. "DPP") that
    happens to sit right next to a redacted name/account field — that
    shape check alone can't tell the two apart, since both are
    capitalized, alphabetic-only words.
    """
    for word in ensemble_words:
        if word.bbox.w <= 0 or word.bbox.h <= 0 or not _is_name_shaped(word, stopwords):
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

        extended_words = [
            w
            for w in ensemble_words
            if _bbox_overlap_fraction(w.bbox, extended_bbox) >= _SPILLOVER_COVERAGE_THRESHOLD
        ]
        multiline = len(_row_clusters(extended_words)) > 1 if extended_words else False

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
            multiline=multiline,
            redaction_id=target.region_id,
        )
        target.engines_seen = sorted(set(target.engines_seen) | set(word.engines))


_LINE_WRAP_ROW_Y_OVERLAP_FRACTION = 0.5
# Caps how much a single word can stretch a _row_clusters envelope beyond
# the row-set's own typical (median) word height. Wide enough that a
# genuinely tall word (an accented capital, a superscript) isn't
# penalized, but tight enough that a mis-segmented OCR box spanning two
# real text lines (often ~2x+ the typical height) can't drag a real next
# line into the same cluster.
_ROW_CLUSTER_MAX_HEIGHT_FACTOR = 1.5
# How much wider the combined union of a match's per-line word clusters
# must be than the clusters' own widths added together before it's treated
# as a genuine line-wrap split rather than compactly co-located text (see
# _line_wrap_clusters). A ratio of exactly 1.0 is the natural dividing
# line: the per-line clusters' own widths summed can only exceed the
# union's width (ratio < 1) when their horizontal extents *overlap* —
# the column-aligned wrapped-cell shape, where every line starts at
# roughly the same x — and can only be smaller than the union (ratio > 1)
# when there's a genuine, uncovered horizontal gap between them, which is
# exactly the tail-of-line-1 / head-of-line-2 shape a real paragraph or
# sentence wrap takes (an ordinary left-margin continuation included, not
# just the "opposite ends" extreme). 1.05 keeps a small buffer above 1.0
# for OCR box jitter without needing the union to be drastically wider
# than the fragments themselves — a real, if modest, gap is already
# proof the words don't compactly co-locate, and painting one box across
# it would otherwise sweep in whatever unrelated text fills that gap on
# both lines (see the regression this guards: a match whose own words
# were "REKSA DANA PENDAPATAN TETAP" at the tail of one line and "BNI AM
# TEAKWOOD KELAS R1" at the head of the next, ratio ~1.53, was falling
# through this gate under the old 1.6 threshold and painting a single
# box spanning nearly the full width of both lines).
_LINE_WRAP_WASTE_RATIO = 1.05


def _row_clusters(words: list[EnsembleWord]) -> list[list[EnsembleWord]]:
    """Group ``words`` into visual-row clusters by vertical (y) overlap.

    A lightweight local clustering pass — not table-aware like
    ``field_extractor``'s ``_group_rows`` — used only to tell whether a
    matched span's own words sit on one visual line or spill across more
    than one (see ``_line_wrap_clusters``). Returned in top-to-bottom
    reading order.

    The overlap check is measured against ``typical_h`` — the *median*
    word height across ``words`` — rather than trusting each word's own
    ``bbox.h``. On a heavily blurred/warped scan, an OCR engine's line
    segmentation can misfire and hand back one word's box tall enough to
    span two physical text lines (see ``native_text``/``ensemble`` box-
    height notes). Two safeguards keep that one inflated box from poisoning
    the whole clustering pass:

    1. The overlap threshold for that word is measured against
       ``typical_h``, not its own bloated ``bbox.h`` — so it still merges
       into the row it visually starts on.
    2. Once merged, its contribution to the cluster's own vertical
       *envelope* is capped at ``_ROW_CLUSTER_MAX_HEIGHT_FACTOR *
       typical_h`` — otherwise the cluster's bottom edge balloons out to
       the bloated word's real bottom, and the next line down then looks
       like it overlaps this (falsely enlarged) envelope too, chain-
       merging a line it never actually touched.

    Getting this wrong feeds ``multiline`` in
    ``_collect_redactions``/``apply_padding``, which relies on this
    function correctly detecting >1 row to suppress top/bottom padding —
    a wrongly-merged two-line value gets the normal per-tier pad added on
    top of its already two-line-tall box instead.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.bbox.y, w.bbox.x))
    heights = sorted(w.bbox.h for w in ordered if w.bbox.h > 0)
    typical_h = heights[len(heights) // 2] if heights else 1
    max_h = typical_h * _ROW_CLUSTER_MAX_HEIGHT_FACTOR

    def _capped_bottom(w: EnsembleWord) -> float:
        return w.bbox.y + min(w.bbox.h, max_h)

    clusters: list[list[EnsembleWord]] = []
    for w in ordered:
        w_bottom = _capped_bottom(w)
        for cluster in clusters:
            cy1 = min(c.bbox.y for c in cluster)
            cy2 = max(_capped_bottom(c) for c in cluster)
            overlap = min(cy2, w_bottom) - max(cy1, w.bbox.y)
            ref_h = min(cy2 - cy1, w_bottom - w.bbox.y, typical_h)
            if overlap > _LINE_WRAP_ROW_Y_OVERLAP_FRACTION * ref_h:
                cluster.append(w)
                break
        else:
            clusters.append([w])
    clusters.sort(key=lambda c: min(x.bbox.y for x in c))
    return clusters


def _line_wrap_clusters(
    matched_words: list[EnsembleWord],
) -> list[list[EnsembleWord]] | None:
    """Per-line word clusters when a matched span's own text wraps across
    a real PDF line break — e.g. a name whose first word(s) sit at the
    tail end of one line and whose remaining word(s) sit at the head of
    the next — rather than the whole span living on a single visual row.

    Painting one union box across both lines in that shape sweeps in
    whatever unrelated text sits between the two clusters' horizontal
    extents: the box's x-range runs from the second line's left-most
    matched word to the first line's right-most one, i.e. nearly the full
    line width on *both* lines, not just the matched words themselves —
    and this isn't only an "opposite ends" extreme: an ordinary
    left-margin paragraph continuation (the tail of one sentence ending
    mid-line, its wrap starting a fresh line at the left margin) takes
    this same shape and sweeps in just as much unrelated text. Detected
    via a "waste ratio": if the combined union of the per-line clusters
    is any wider than the clusters' own widths added together, there's a
    genuine, uncovered horizontal gap between them — the words don't
    compactly co-locate — and the caller should paint one tight box per
    line instead of a single union box (see ``_collect_redactions``,
    which paints the mock value on only one of the resulting boxes and a
    blank white patch on the rest).

    Returns ``None`` when there's only one visual line, or when the
    clusters' own horizontal extents overlap rather than leaving a gap —
    e.g. a wrapped table cell's lines, which stay column-aligned (every
    line starting at roughly the same x) rather than leaving daylight
    between them, are correctly left as a single box.
    """
    clusters = _row_clusters(matched_words)
    if len(clusters) < 2:
        return None
    boxes = [b for b in (union_bbox(c) for c in clusters) if b is not None]
    if len(boxes) < 2:
        return None
    combined_w = sum(b.w for b in boxes)
    union_x1 = min(b.x for b in boxes)
    union_x2 = max(b.x + b.w for b in boxes)
    union_w = union_x2 - union_x1
    if combined_w <= 0 or union_w <= combined_w * _LINE_WRAP_WASTE_RATIO:
        return None
    return clusters


def _digital_native_line_wrap_clusters(
    matched_words: list[EnsembleWord],
) -> list[list[EnsembleWord]] | None:
    """Digital-page-only fallback for a match ``_line_wrap_clusters``
    itself leaves unsplit because every line starts at the same left
    margin — the normal shape for a deliberately multi-line field value
    (e.g. a wrapped postal address split "MFAR GREEN HEART, PHASE IV" /
    "MFAR-MANYATA TECH PARK" across two real PDF lines), as opposed to
    an accidentally-wrapped prose sentence whose *tail* lands mid-line.
    ``_line_wrap_clusters``'s waste-ratio gate treats that shape the
    same as a wrapped table cell's column-aligned lines (deliberately —
    see its own docstring) and keeps it as one box, which still paints a
    single, oversized patch with the mock value floating, vertically
    centered, across both lines instead of tight per-line boxes.

    Only ever called for a digital (native-PDF-text) page — never a
    scanned/OCR one. ``_row_clusters``'s row split is trustworthy
    outright here: ``native_text.py`` assigns every word on a line the
    PDF's own exact ``(block_no, line_no)`` bbox straight from the
    content stream, not an OCR-inferred one, so there's none of the
    noisy-row-detection ambiguity ``_line_wrap_clusters``'s gate exists
    to guard scanned pages against (see ``_row_clusters``'s docstring).
    Scanned-page behavior (and its regression tests) is entirely
    untouched — this is only ever consulted as an *additional* fallback
    when ``_line_wrap_clusters`` already returned ``None``, and only on
    a digital page (see ``_collect_redactions``).

    Returns ``None`` when there's only one visual line (nothing to
    split), else every visual row ``_row_clusters`` finds.
    """
    clusters = _row_clusters(matched_words)
    if len(clusters) < 2:
        return None
    return clusters


_BRAND_ZONE_ENTITY_TYPES = {
    "footer": "BRAND_FOOTER",
    "picture": "BRAND_IMAGE",
    "signature": "SIGNATURE",
}


def _brand_region(zone: BrandZone, blur_tier: str, region_id: str) -> RedactionRegion:
    """Map BrandZone → RedactionRegion. Dummy scores; no dictionary row."""
    entity_type = _BRAND_ZONE_ENTITY_TYPES[zone.zone]
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
        ocr_output_store: OcrOutputStore | None = None,
    ) -> None:
        from app.api.mock_routes import get_ledger_store, get_mock_store, get_ocr_output_store

        self.settings = settings or get_settings()
        self.mock_store = mock_store or get_mock_store()
        self.ledger_store = ledger_store or get_ledger_store()
        self.audit_store = audit_store or AuditStore()
        self.ocr_output_store = ocr_output_store or get_ocr_output_store()

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
        self._dump_ocr_output(request_id, filename, page_states)
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

    def _dump_ocr_output(
        self,
        request_id: str,
        filename: str,
        page_states: list[PageProcessState],
    ) -> None:
        """Best-effort diagnostic dump (Settings.ocr_output_dump_enabled) —
        see app/services/redact/ocr_output_store.py. Only written on a
        fresh OCR run (``run()``, not ``regenerate()``, which reuses
        already-cached page_states with nothing new to dump). Never lets a
        write failure (e.g. a disk-full machine) fail the actual redaction
        request — this file only exists to help diagnose OCR/fuzzy-match
        issues after the fact.
        """
        if not self.settings.ocr_output_dump_enabled:
            return
        try:
            self.ocr_output_store.save(request_id, filename, page_states)
        except Exception:
            logger.warning(
                "ocr_output_dump failed for request_id=%s", request_id, exc_info=True
            )

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

        # Classify each page (digital vs. scanned) from its native
        # fitz.Page — before canonicalizing/deskewing — so a digital page
        # can skip deskewing entirely: a real vector-text PDF page is never
        # skewed in practice, and skipping it avoids having to re-project
        # native word boxes through a rotation matrix. classify_and_extract
        # never raises (it fails safe to "scanned" internally), and
        # rendered.fitz_page is already None for non-PDF input and the
        # pdf2image fallback path, both of which are always "scanned".
        page_kinds: list[str] = []
        native_results: list[tuple[str, list[EnsembleWord]]] = []
        for idx, rendered in enumerate(pages):
            if self.settings.native_text_bypass_enabled:
                kind, native_merged, native_words = classify_and_extract(
                    rendered.fitz_page,
                    opts.dpi,
                    idx,
                    self.settings.native_text_min_words,
                    self.settings.native_text_min_coverage_pct,
                )
            else:
                kind, native_merged, native_words = "scanned", "", []
            page_kinds.append(kind)
            native_results.append((native_merged, native_words))

        canonical_pages: list[CanonicalPage] = []
        try:
            for idx, rendered in enumerate(pages):
                canonical_pages.append(
                    canonicalize_page(
                        rendered.image,
                        idx,
                        strip_gridlines=self.settings.strip_gridlines_enabled,
                        deskew=self.settings.deskew_enabled and page_kinds[idx] != "digital",
                    )
                )
        except Exception as exc:
            raise PipelineStageError("preprocess", "Preprocessing failed", exc) from exc

        sample_text = ""
        locale, langs, tess_lang = resolve_languages(opts.locale, opts.languages, opts.auto_detect, sample_text)
        page_states: list[PageProcessState] = []

        for idx, canonical in enumerate(canonical_pages):
            page_kind = page_kinds[idx]
            native_merged, native_words = native_results[idx]
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

            ocr_engines_used: list[str] = []
            if page_kind == "digital":
                # Native PDF text already extracted above (classify_and_extract) —
                # skip OCR entirely for this page.
                merged_text, ensemble_words = native_merged, native_words
            else:
                try:
                    merged_text, ensemble_words, engine_results = await ensemble_ocr_page(
                        canonical.canonical_image,
                        canonical.page_index,
                        tess_lang,
                        langs,
                        table_regions=table_regions,
                        engine_filter=opts.ocr_engines,
                    )
                    ocr_engines_used = [r.engine for r in engine_results]
                except Exception as exc:
                    # Every configured OCR engine failed/returned nothing for
                    # this page (e.g. a low-contrast scan, or a "hybrid"
                    # PDF whose embedded text layer didn't clear the
                    # digital-page thresholds above). Rather than aborting
                    # the whole document, fall back to whatever native PDF
                    # text this page actually has — a page with *some*
                    # copyable text is strictly better redacted from that
                    # text than not redacted at all. Only a page with no
                    # text layer at all (fitz_page is None, or truly
                    # scanned-image-only) still raises.
                    fallback_merged, fallback_words = "", []
                    if self.settings.native_text_bypass_enabled and pages[idx].fitz_page is not None:
                        fallback_merged, fallback_words = extract_native_words(
                            pages[idx].fitz_page, opts.dpi, idx
                        )
                    if fallback_words:
                        logger.warning(
                            "ensemble_ocr failed for page_index=%s (%s); falling back to "
                            "native PDF text layer",
                            idx,
                            type(exc).__name__,
                        )
                        merged_text, ensemble_words = fallback_merged, fallback_words
                        page_kind = "digital"
                    elif self.settings.ocr_blank_page_skip_enabled:
                        # No OCR text AND no native text layer at all —
                        # either a genuinely blank page (disclaimer/back
                        # page) or a scan too degraded for every engine to
                        # read; "zero words" alone can't distinguish those.
                        # Skipping keeps one such page from blocking the
                        # rest of a multi-page document — logged loudly
                        # (not silent) since 0 redactions is possible on
                        # this page either way. Disable via
                        # Settings.ocr_blank_page_skip_enabled if this
                        # document type must never ship a page OCR
                        # couldn't read.
                        logger.warning(
                            "page_index=%s: no OCR text and no native text layer (%s) — "
                            "treating as blank/unreadable, 0 redactions possible on this "
                            "page; set ocr_blank_page_skip_enabled=False to fail the whole "
                            "document instead",
                            idx,
                            type(exc).__name__,
                        )
                        merged_text, ensemble_words = "", []
                        page_kind = "blank"
                    else:
                        raise PipelineStageError(
                            "ensemble_ocr",
                            f"OCR ensemble failed on page_index={idx} ({type(exc).__name__}); "
                            "see preceding app.services.ocr.ensemble log lines for the "
                            "per-engine breakdown",
                            exc,
                        ) from exc

            sample_text += merged_text + "\n"
            if opts.auto_detect and canonical.page_index == 0 and not opts.locale:
                locale, langs, tess_lang = resolve_languages(None, None, True, sample_text)

            try:
                word_context = join_words_to_blocks(ensemble_words, blocks)
            except Exception as exc:
                raise PipelineStageError("docling", "Structure extraction failed", exc) from exc

            # One consolidated line tying together every input to the
            # padding-geometry decision (apply_padding, further downstream)
            # that's otherwise scattered across separate log lines from
            # canonicalize_page/extract_structure/ensemble_ocr_page — the
            # fastest way to diff "why did this page redact differently on
            # machine A vs. machine B" without re-running the visualization
            # script or comparing screenshots.
            logger.info(
                "page processed",
                extra={
                    "page_index": idx,
                    "page_kind": page_kind,
                    "blur_tier": canonical.transform.blur_tier,
                    "blur_variance": canonical.transform.blur_variance,
                    "skew_angle_deg": canonical.transform.skew_angle_deg,
                    "structure_block_count": len(blocks),
                    "structure_block_type_counts": dict(
                        Counter(b.block_type for b in blocks)
                    ),
                    "ocr_word_count": len(ensemble_words),
                    "ocr_engines_used": ocr_engines_used,
                },
            )

            page_states.append(
                PageProcessState(
                    canonical=canonical,
                    merged_text=merged_text,
                    ensemble_words=ensemble_words,
                    blocks=blocks,
                    word_context=word_context,
                    page_kind=page_kind,
                    fitz_page=pages[idx].fitz_page,
                    dpi=opts.dpi,
                    _fitz_doc=pages[idx]._doc,
                )
            )

        return page_states

    def _spillover_stopwords(self) -> frozenset[str]:
        """Parsed, upper-cased ``Settings.spillover_non_name_stopwords``."""
        return frozenset(
            word.strip().upper()
            for word in self.settings.spillover_non_name_stopwords.split(",")
            if word.strip()
        )

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
        all_redactions: list[RedactionRegion] = []
        page_audits: list[PageAuditSummary] = []
        ledger_rows: list[dict] = []
        brand_zone_dicts: list[dict] = []
        region_counter = region_start
        # One physical patch per underlying text: field-anchored,
        # dictionary-scan, and custom terms can independently land on the
        # same words. Exact-coordinate keys miss near-duplicates (a cell
        # box vs. a slightly expanded union), so overlap is checked
        # against already-accepted page_redactions instead.

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

            for term in opts.custom_redactions:
                for start, end in find_term_spans(merged_text, term.search_value):
                    spans.append(_SpanCandidate(start, end, "CUSTOM", 0.95, term, None, None))

            if self.settings.dictionary_scan_enabled:
                # Every value already curated in the mock dictionary is
                # searched for directly in this page's OCR text, the same
                # way a per-request custom term is (find_term_spans) —
                # independent of field_extractor's layout/label detection,
                # so a known name/org is caught anywhere it appears on the
                # page (e.g. inside a transaction narration, not just an
                # "A/C Name:" field) even on a document shape field_extractor
                # doesn't recognize. Can only ever match text that's already
                # known; it never discovers brand-new unseen PII on its own.
                # Longest match wins: a short alias ("Standard Chartered",
                # "SCB") nested inside a longer curated name is skipped so
                # we don't paint two overlapping patches. Custom-term char
                # ranges are claimed first; field-anchored geometry is
                # de-duplicated later via padded-box overlap because a
                # table cell's words are often non-contiguous in
                # merged_text.
                claimed_char_ranges = [
                    (span.start, span.end) for span in spans if not span.words
                ]
                for entry in sorted(
                    self.mock_store.list(),
                    key=lambda e: len(e.source_text),
                    reverse=True,
                ):
                    for start, end in find_term_spans(merged_text, entry.source_text):
                        if _char_ranges_overlap(start, end, claimed_char_ranges):
                            continue
                        claimed_char_ranges.append((start, end))
                        spans.append(
                            _SpanCandidate(start, end, "KNOWN_TERM", 0.95, None, None, None)
                        )

                if self.settings.fuzzy_dictionary_scan_enabled:
                    # A single OCR-garbled character (misread letter, a
                    # garbled/replacement glyph) breaks the exact scan above
                    # entirely. Probed for every entry — not only ones with
                    # zero exact hits — because the same entry can appear on
                    # a page both cleanly (several exact-matched table rows)
                    # and once garbled (e.g. a signature block).
                    # find_fuzzy_term_spans itself already looks for several
                    # distinct approximate occurrences per entry, but a
                    # clean occurrence would still win every one of those
                    # slots (it scores highest) before a different,
                    # still-uncaught garbled one ever got a turn — masking
                    # out every already-claimed range first (see
                    # _mask_claimed_ranges) removes the clean occurrences
                    # from contention entirely, so the whole per-entry probe
                    # budget goes to text nothing has matched yet.
                    masked_text = _mask_claimed_ranges(merged_text, claimed_char_ranges)
                    fuzzy_candidates: list[tuple[float, int, int]] = []
                    for entry in self.mock_store.list():
                        for start, end in find_fuzzy_term_spans(
                            masked_text,
                            entry.source_text,
                            threshold=self.settings.fuzzy_dictionary_scan_threshold,
                        ):
                            # Score from the actual matched text's
                            # similarity, not a fixed constant, so a
                            # marginal fuzzy hit scores lower confidence
                            # than a clean exact one downstream.
                            fuzzy_score = token_sort_ratio(
                                normalize_source(entry.source_text),
                                normalize_source(merged_text[start:end]),
                            )
                            fuzzy_candidates.append((fuzzy_score, start, end))
                else:
                    fuzzy_candidates = []

                # Best-scoring fuzzy match wins a contested range, not the
                # longest dictionary entry: a longer entry compared against
                # a short garbled OCR fragment can still clear the threshold
                # by aligning against a window padded with unrelated
                # surrounding characters (fuzz.partial_ratio_alignment picks
                # whatever substring scores best, not necessarily a
                # semantically real one) — letting length decide first (as
                # the exact pass above safely does) would let that
                # coincidental, lower-confidence match shadow a shorter,
                # far more accurate one for the same span.
                for fuzzy_score, start, end in sorted(
                    fuzzy_candidates, key=lambda c: c[0], reverse=True
                ):
                    if _char_ranges_overlap(start, end, claimed_char_ranges):
                        continue
                    claimed_char_ranges.append((start, end))
                    spans.append(
                        _SpanCandidate(
                            start, end, "KNOWN_TERM_FUZZY", fuzzy_score, None, None, None
                        )
                    )

            for start, end, entity_type, score, term, _field_role, _account_number, cand_words in spans:
                # Joins this redaction's "span mapped to bbox" / "bbox
                # padded and clamped" / "redaction scored" / "mock_resolve"
                # debug lines (otherwise scattered/interleaved across
                # modules with every other redaction on the page) into one
                # traceable identity — deterministic from the candidate's
                # own char span, no extra state needed.
                redaction_id = f"p{canonical.page_index}:{start}-{end}"
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
                    bbox = map_span_to_ensemble_bbox(
                        start, end, ensemble_words, merged_text, redaction_id=redaction_id
                    )
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
                    # _resolve_maximal_munch_window). find_prefix_collisions
                    # itself excludes a deliberate base+suffix family member
                    # (e.g. "DPLK AXA MANDIRI" vs. "... - PPIP PU") from
                    # this signal, so this probe is never triggered for one.
                    normalized_candidate = normalize_source(source_text)
                    if normalized_candidate and self.mock_store.find_prefix_collisions(
                        normalized_candidate
                    ):
                        matched_words, source_text = _resolve_maximal_munch_window(
                            matched_words,
                            ensemble_words,
                            self.mock_store,
                            blocks=state.blocks,
                        )
                        bbox = union_bbox(matched_words)

                word_bboxes = [w.bbox for w in matched_words] if matched_words else None
                cell_bbox = _enclosing_cell_bbox(bbox, state.blocks, word_bboxes=word_bboxes)

                # A merged_text char-range match (the ``else`` branch above,
                # as opposed to a field-anchored ``cand_words`` candidate)
                # can pick up a word from an unrelated block when the
                # ensemble's reading order glitches at a row/column
                # boundary (see ensemble.py's _reading_order docstring) —
                # words_for_span/union_bbox has no spatial awareness, so
                # the resulting union silently spans both the real match
                # and the stray word's actual location. apply_padding's
                # cell-clamp never shrinks below the union's own height,
                # so an inflated union here would otherwise paint past the
                # cell into whatever sits above/below it.
                if cell_bbox is not None and not cand_words:
                    filtered = _filter_words_to_cell(matched_words, cell_bbox)
                    if filtered is not None:
                        matched_words = filtered
                        bbox = union_bbox(matched_words)
                        source_text = " ".join(w.text for w in matched_words)

                # A non-tabular match whose own words wrap across a real
                # PDF line break (some words at the tail of one line, the
                # rest at the head of the next) is painted as one tight
                # box per line instead of a single union box spanning
                # nearly the full width of both lines — see
                # _line_wrap_clusters. Table-cell matches are excluded
                # (cell_bbox is not None): a wrapped cell's lines stay
                # column-aligned and are already handled as one patch.
                line_clusters = (
                    _line_wrap_clusters(matched_words)
                    if cell_bbox is None and matched_words
                    else None
                )
                if (
                    line_clusters is None
                    and cell_bbox is None
                    and matched_words
                    and state.page_kind == "digital"
                ):
                    # Digital-only fallback — see
                    # _digital_native_line_wrap_clusters. Never runs for
                    # a scanned page: state.page_kind gates it strictly,
                    # so existing OCR line-wrap behavior is unchanged.
                    line_clusters = _digital_native_line_wrap_clusters(matched_words)
                if line_clusters is not None:
                    candidate_boxes = [
                        (cluster_bbox, cluster)
                        for cluster in line_clusters
                        if (cluster_bbox := union_bbox(cluster)) is not None
                    ]
                else:
                    candidate_boxes = [(bbox, matched_words)]

                painted: list[tuple[BBox, BBox, BBox, list[EnsembleWord]]] = []
                for sub_bbox, sub_words in candidate_boxes:
                    sub_original_bbox = canonical_to_original(sub_bbox, canonical.transform)
                    sub_word_bboxes = [w.bbox for w in sub_words] if sub_words else None
                    sub_cell_bbox = cell_bbox or _enclosing_cell_bbox(
                        sub_bbox, state.blocks, word_bboxes=sub_word_bboxes
                    )
                    left_neighbor_x = right_neighbor_x = None
                    original_cell_bbox = None
                    if sub_cell_bbox is not None:
                        original_cell_bbox = canonical_to_original(sub_cell_bbox, canonical.transform)
                    else:
                        excluded_ids = {id(w) for w in sub_words}
                        left_x, right_x = _row_neighbor_clamp_x(sub_bbox, ensemble_words, excluded_ids)
                        dx = canonical.transform.dx
                        left_neighbor_x = left_x + dx if left_x is not None else None
                        right_neighbor_x = right_x + dx if right_x is not None else None
                    # A match whose own words already span more than one
                    # visual line (e.g. a wrapped table-cell value) has its
                    # full, correct vertical extent in sub_bbox already —
                    # adding the usual top/bottom padding on top of that
                    # risks bleeding into a tightly-packed neighboring row.
                    # A per-line box carved out of line_clusters is, on its
                    # own, single-line — but it's still only a fragment of
                    # an originally multi-line match sitting immediately
                    # above/below its sibling line box, so it needs the
                    # same padding suppression: a tightly-leaded document
                    # (e.g. a digital bank statement) can pack the next
                    # real line close enough that even the small per-tier
                    # pad reaches into it, and vector-native redaction
                    # (pdf_native_redactor.py) genuinely deletes whatever
                    # text its rect touches — not just visually covers it.
                    sub_multiline = line_clusters is not None or (
                        len(_row_clusters(sub_words)) > 1 if sub_words else False
                    )
                    sub_padded = apply_padding(
                        sub_original_bbox,
                        canonical.transform.blur_tier,
                        canonical.original_image.width,
                        canonical.original_image.height,
                        cell_bbox=original_cell_bbox,
                        left_neighbor_x=left_neighbor_x,
                        right_neighbor_x=right_neighbor_x,
                        multiline=sub_multiline,
                        redaction_id=redaction_id,
                    )
                    if _padded_overlaps_existing(sub_padded, page_redactions):
                        continue
                    painted.append((sub_bbox, sub_original_bbox, sub_padded, sub_words))

                if not painted:
                    continue

                if not source_text.strip():
                    continue

                user_mock = _user_mock_for_term(term) if term is not None else None
                if term is None and self.settings.restrict_to_known_mappings:
                    # Field-anchored/dictionary-scan candidate under
                    # restricted mode: only paint it if it already matches
                    # the curated dictionary (exact or trusted fuzzy) —
                    # never auto-create a new entry for unseen text (moot
                    # for dictionary-scan hits, which are already known by
                    # construction). Explicit custom redaction terms (term
                    # is not None) are a direct user instruction and always
                    # resolve/create regardless.
                    entry = self.mock_store.lookup(source_text)
                    if entry is None and cand_words:
                        # A field-anchored candidate (cand_words truthy —
                        # e.g. a "Nama Rekening" table-cell value) already
                        # carries its own layout-based confidence that this
                        # span *is* PII, independent of exact spelling.
                        # restrict_to_known_mappings exists to stop new,
                        # auto-created dictionary rows for unseen text (see
                        # Settings.restrict_to_known_mappings) — it isn't
                        # meant to let an already-known client name leak
                        # through fully unredacted just because a couple of
                        # OCR-garbled characters push it under lookup()'s
                        # trusted-fuzzy bar (e.g. "PT 8NILIFE INSURANCE" for
                        # "PT BNI LIFE INSURANCE"). best_unambiguous_match
                        # only hands back a match when it's decisively
                        # ahead of every other, unrelated entry, so this
                        # never risks painting the wrong client's mock over
                        # a merely-similar one.
                        normalized_fallback = normalize_source(source_text)
                        entry, _ratio = self.mock_store.best_unambiguous_match(normalized_fallback)
                    if entry is None:
                        continue
                else:
                    entry = self.mock_store.resolve(source_text, user_mock=user_mock)
                logger.info(
                    "mock_resolve mapping_id=%s entity_type=%s assignment_source=%s",
                    entry.mapping_id,
                    entity_type,
                    entry.assignment_source,
                    extra={"redaction_id": redaction_id},
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

                confidence, breakdown = score_redaction(
                    score, matched_words, ctx, redaction_id=redaction_id
                )
                engines_seen = sorted({e for w in matched_words for e in w.engines})
                # When the match was split across a line wrap, only the
                # cluster carrying the most of the actual matched text
                # gets the mock value drawn on it — the other line(s) are
                # painted as blank white patches (mock_value=""), per
                # _line_wrap_clusters: splitting the mock text itself
                # across two disjoint boxes has no sensible single font
                # fit and would read as two separate values.
                primary_idx = max(
                    range(len(painted)),
                    key=lambda i: sum(len(w.text) for w in painted[i][3]),
                )
                for idx, (sub_bbox, sub_original_bbox, sub_padded, _sub_words) in enumerate(painted):
                    region_counter += 1
                    page_redactions.append(
                        RedactionRegion(
                            region_id=f"r-{region_counter:04d}",
                            page=canonical.page_index,
                            entity_type=entity_type,
                            canonical_bbox=sub_bbox,
                            original_bbox=sub_original_bbox,
                            padded_bbox=sub_padded,
                            redaction_confidence=confidence,
                            confidence_breakdown=breakdown,
                            structural_context=ctx,
                            blur_tier=canonical.transform.blur_tier,
                            engines_seen=engines_seen,
                            mock_value=entry.mock_value if idx == primary_idx else "",
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
                _apply_spillover_safety_net(
                    page_redactions,
                    ensemble_words,
                    state.blocks,
                    canonical,
                    self._spillover_stopwords(),
                )

            original = canonical.original_image
            zones = detect_brand_zones(
                page_w=original.width,
                page_h=original.height,
                page=canonical.page_index,
                blocks=state.blocks,
                patch_footer=opts.patch_footer,
                footer_bottom_pct=self.settings.footer_zone_bottom_pct,
            )
            zones = zones + detect_picture_zones(
                page_w=original.width,
                page_h=original.height,
                page=canonical.page_index,
                blocks=state.blocks,
                existing_zones=zones,
                enabled=opts.patch_images and self.settings.patch_images_enabled,
                min_area_pct=self.settings.image_zone_min_area_pct,
            )
            zones = zones + detect_signature_zones(
                ensemble_words,
                page_w=original.width,
                page_h=original.height,
                page=canonical.page_index,
                existing_zones=zones,
                enabled=opts.patch_signatures and self.settings.patch_signatures_enabled,
                fitz_page=state.fitz_page if state.page_kind == "digital" else None,
                dpi=state.dpi,
            )
            text_boxes = [r.padded_bbox for r in page_redactions]
            min_picture_area = (
                self.settings.image_zone_min_area_pct * original.width * original.height
            )
            zones = reconcile_picture_zones_with_text(
                zones, text_boxes, min_area=min_picture_area
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
                    page_kind=state.page_kind,
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
            page_inputs = [
                PageRenderInput(
                    image=s.canonical.original_image,
                    page_kind=s.page_kind,
                    fitz_page=s.fitz_page,
                    dpi=s.dpi,
                )
                for s in page_states
            ]
            return render_redacted_pdf(
                page_inputs,
                redactions,
                filename,
                image_format=self.settings.redact_output_image_format,
                jpeg_quality=self.settings.redact_output_jpeg_quality,
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
