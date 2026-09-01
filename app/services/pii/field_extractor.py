"""Field-anchored PII detection: labels, table columns, prose, signatures.

Replaces whole-page NER (Presidio/spaCy), which hallucinates PERSON/ORG
entities on noisy OCR text, with detection anchored to document structure:
a label immediately before its value (``A/C Name :``), a table column
identified by its header (``Nama Rekening``), a prose ``a/n`` marker, or a
short Title-Case line in the bottom signature block.

Detection runs primarily on ``EnsembleWord`` geometry (bbox + char offsets
already aligned to ``merged_text``), not on ``DocBlock``/Docling output —
this keeps it working even when Docling is unavailable or degraded. The one
exception: when Docling ``cell`` blocks are supplied, their bounding boxes
are used purely as a *row-merge signal* (see ``_stitch_docling_cells``) to
recombine a multi-line-wrapped table cell that pure y-adjacency row
clustering split into separate rows — Docling's own table-structure model
recognizes a wrapped cell as a single cell spanning all its lines, which
``_group_rows`` has no way to represent on its own. Docling's column
*labels* are never trusted here (frequently ``None`` or OCR-garbled) —
column identity still comes entirely from this module's own header-fuzzy-
matching and x-position zone logic. ``word_context`` is accepted for
interface stability only and is not used.

Every function here only ever emits a candidate for a *name-shaped* value:
number-only, date-only, and account-number-shaped text is filtered out, so
account numbers and dates are never accidentally redacted even if a label
match misfires (SEC-002 scope guard).

Never logs candidate or source text (SEC-001) — only counts and roles.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

from app.models.pii_chunk import BBox
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.pii import field_labels
from app.services.pii.ensemble_mapper import union_bbox
from app.services.pii.name_matcher import best_fuzzy_match, token_sort_ratio
from app.services.structure.docling_adapter import DocBlock

logger = logging.getLogger(__name__)

_WORD_TOKEN_RE = re.compile(r"\S+")
_SPLIT_TOKEN_PAD_PX = 10

# Relative glyph-width table for estimating a sub-token's split point inside
# a merged line-level OCR box (see _split_multiword_tokens). Values are
# rough ratios to an average lowercase letter (1.0), loosely modeled on
# common proportional-font metrics (e.g. Helvetica/Arial advance widths):
# digits are all a fixed, slightly-narrower-than-average width in most
# fonts; punctuation/thin letters (i, l, punctuation, space) are narrow;
# M/W-class letters and uppercase generally run wider. This is only ever
# an estimate to seed the *initial* split point — the interior-boundary
# midpoint clamp below still prevents adjacent sub-tokens from overlapping
# regardless of how accurate the estimate is.
_DIGIT_CHAR_WIDTH = 0.85
_NARROW_CHAR_WIDTH = 0.45
_WIDE_CHAR_WIDTH = 1.5
_DEFAULT_CHAR_WIDTH = 1.0
_NARROW_CHARS = frozenset(" \t.,:;'\"|!()[]{}ijl")
_WIDE_CHARS = frozenset("MWmw@")


def _char_width(ch: str) -> float:
    """Relative width estimate for one character (see _CHAR_WIDTH table docs)."""
    if ch.isdigit():
        return _DIGIT_CHAR_WIDTH
    if ch in _NARROW_CHARS:
        return _NARROW_CHAR_WIDTH
    if ch in _WIDE_CHARS:
        return _WIDE_CHAR_WIDTH
    return _DEFAULT_CHAR_WIDTH


def _cumulative_char_widths(text: str) -> list[float]:
    """Cumulative relative width up to (and including) each character.

    ``result[i]`` is the total estimated width of ``text[:i]``, so a
    character span's fractional x-position within the box is
    ``result[i] / result[-1]`` — replacing a uniform per-character-count
    fraction with one that accounts for digits/narrow/wide glyphs running
    narrower or wider than an average letter. Includes whitespace between
    tokens so gaps between words also consume their estimated share of the
    box width, rather than being silently absorbed into a neighboring
    token's span.
    """
    cumulative = [0.0]
    total = 0.0
    for ch in text:
        total += _char_width(ch)
        cumulative.append(total)
    return cumulative

_MAX_LABEL_WINDOW = 6
# See the "stolen token" guard in _find_all_label_windows_multi: how similar
# a single token must be to another group's single-word label before a
# longer window is barred from consuming it. Deliberately looser than any
# accept threshold (0.72+) — this only needs to recognize "this token wants
# to be its own thing", not confirm a full match.
_STOLEN_TOKEN_THRESHOLD = 0.7
_ROW_Y_TOLERANCE_FLOOR = 4.0
_ROW_Y_TOLERANCE_FACTOR = 0.6
# Wider than _ROW_Y_TOLERANCE_FACTOR, used only by the column-band fallback
# (_column_band_stitch) below, and only between two rows already confirmed
# to share exactly one recognized table column — normal multi-line cell
# wrapping (e.g. a bank name wrapping onto its own next line) has a
# line-to-line gap noticeably larger than the tight same-line word spacing
# _group_rows tunes for, but is still much closer than genuinely unrelated
# content below the table.
_COLUMN_BAND_Y_TOLERANCE_FACTOR = 2.2

_SECTION_MAX_ROW_WORDS = 6
_ADDRESSEE_MAX_ROW_WORDS = 4
_NUMBER_PAIR_MAX_ROW_DISTANCE = 3

_AN_MAX_VALUE_WORDS = 8
_AN_LOOKBACK_WORDS = 15

_KEPADA_MAX_VALUE_WORDS = 8
_KEPADA_STOP_WORDS = ("untuk", "guna", "agar", "supaya")

_SIGNATURE_BOTTOM_FRACTION = 0.3
_SIGNATURE_MIN_WORDS = 2
_SIGNATURE_MAX_WORDS = 4
_SIGNATURE_ORG_MIN_WORDS = 2
_SIGNATURE_ORG_MAX_WORDS = 8

# A dual-column bank table header (Jumlah IDR/USD, Debit/Kredit Rekening x
# Nama/Bank/No each) already runs ~11 labels wide; the OCR ensemble commonly
# emits each header token twice (once per engine, at slightly different x
# positions) before words are deduped downstream, so real header rows can
# run 25-30 tokens. 40 leaves headroom for that duplication while still
# rejecting genuine paragraph-length content rows.
_HEADER_MAX_ROW_WORDS = 40

_BASE_ROLE_ACCOUNT_NAME = "name"
_BASE_ROLE_BANK = "bank"
_BASE_ROLE_ACCOUNT_NUMBER = "number"

_NUMERIC_RE = re.compile(r"^[\d\s\-.,/:]+$")

# Common OCR letter/digit confusions folded before header/column-*label*
# fuzzy matching only (see _normalize_label_candidate) — never applied to
# redacted value text.
_OCR_CONFUSION_TABLE = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "|": "l"})


@dataclass
class FieldCandidate:
    """A name-shaped span anchored to document structure.

    ``start``/``end`` are the min/max character offsets spanned by ``words``
    — approximate positional metadata only (e.g. for sorting), *not* a
    reliable contiguous range: a table cell's words are visually adjacent
    but commonly non-contiguous in the page's merged-text character stream
    (multi-line cell text interleaves with neighboring columns in whatever
    order the OCR ensemble emitted words). Callers must build geometry and
    source text from ``words`` directly rather than slicing ``merged_text``
    or re-deriving words via character-overlap, which would silently sweep
    in unrelated words that merely fall inside that wide min/max range.
    """

    start: int
    end: int
    entity_type: str
    field_role: field_labels.FieldRole
    match_confidence: float
    words: tuple[EnsembleWord, ...]
    account_number: str | None = None


def _looks_numeric_or_date(text: str) -> bool:
    """True for empty, purely numeric/punctuation, or date-shaped text."""
    stripped = text.strip()
    if not stripped:
        return True
    if _NUMERIC_RE.match(stripped):
        return True
    lowered = stripped.lower()
    has_digit = any(ch.isdigit() for ch in stripped)
    if has_digit and any(month in lowered for month in field_labels.DATE_MONTH_WORDS):
        return True
    return False


def _normalize_label_candidate(text: str) -> str:
    """Fold common OCR letter/digit confusions before header-label fuzzy matching.

    The dual-OCR-engine ensemble's word-cluster tie-break
    (``app/services/ocr/ensemble.py``'s ``align_word_boxes``) picks one
    engine's reading of a header glyph over the other whenever they
    disagree — e.g. "Rekening" misread as "Rekenlng" (1/l) or "Bank" as
    "Bahk" — which of the two readings wins is itself non-deterministic
    across engine availability/versions. Folding well-known OCR
    letter/digit lookalikes to one canonical spelling before comparing
    against the fixed label vocabulary makes header/column matching less
    sensitive to that tie-break's outcome.

    Deliberately scoped to header/column *label* text only (compared
    against a small, known vocabulary in ``field_labels.py``) — a token
    that's already mostly digits is left untouched so a genuine
    account-number-shaped value is never coerced into looking label-like
    (SEC-002 scope guard: this never touches redacted value text).
    """
    stripped = text.strip(" :.-").lower()
    if not stripped:
        return stripped
    digit_count = sum(ch.isdigit() for ch in stripped)
    if digit_count > len(stripped) / 2:
        return stripped
    return stripped.translate(_OCR_CONFUSION_TABLE)


def _split_multiword_tokens(words: list[EnsembleWord]) -> list[EnsembleWord]:
    """Split OCR detections that bundle multiple words into one box.

    Line/phrase-level engines (e.g. EasyOCR) commonly return one bounding
    box per detected text line rather than per word — ``"Nama Rekening"``
    or ``"0.00 PT BNI LIFE INSURANCE"`` as a single token. Every matcher
    below assumes one token per real word, as word-level engines
    (Tesseract/PaddleOCR) already give, so re-tokenize on whitespace here
    and distribute each box proportionally by character position.

    The split point for each sub-token uses a relative digit/letter width
    table (see ``_char_width``) rather than a uniform character-count
    fraction of the box width, so a short numeric amount glued to a long
    name on one OCR line (``"0.00 PT BNI LIFE INSURANCE"``, seen in
    several ``repii/`` samples) gets a boundary closer to its true glyph
    width instead of one skewed by naively assuming every character is the
    same width.

    Sub-token char offsets stay valid subranges of the source word's own
    span, so downstream span -> bbox mapping (which overlaps against the
    original, unsplit ``ensemble_words``) still resolves to that word's
    whole box — a safe, over-inclusive paint rather than a missed one.

    Padding at the *outer* edges (before the first sub-token, after the
    last) is free to use the full ``_SPLIT_TOKEN_PAD_PX`` since there's no
    sibling there to collide with. An *interior* boundary between two
    sub-tokens of the same merged detection is different: naively padding
    both sides by the full amount lets them overlap each other by up to
    ``2 * _SPLIT_TOKEN_PAD_PX`` — harmless when both sub-tokens end up in
    the same redaction candidate, but a real bug when they don't (e.g. a
    non-PII amount glued to a PII name on one OCR line, "0.00 Some Fund" —
    the name's redaction box would then bleed into and paint over part of
    the amount). So each interior boundary is clamped to the midpoint
    between the two sub-tokens' unpadded (proportional-split) edges: both
    sides can still expand toward each other, but never past where the
    other one's own unpadded text was estimated to start/end.
    """
    result: list[EnsembleWord] = []
    for word in words:
        text = word.text
        matches = list(_WORD_TOKEN_RE.finditer(text))
        if len(matches) <= 1:
            result.append(word)
            continue
        cumulative_widths = _cumulative_char_widths(text)
        total_width = cumulative_widths[-1] or 1.0
        raw_spans = []
        for match in matches:
            frac_start = cumulative_widths[match.start()] / total_width
            frac_end = cumulative_widths[match.end()] / total_width
            raw_x = word.bbox.x + round(frac_start * word.bbox.w)
            raw_end = word.bbox.x + round(frac_end * word.bbox.w)
            raw_spans.append((raw_x, raw_end))

        for i, match in enumerate(matches):
            raw_x, raw_end = raw_spans[i]
            left_limit = word.bbox.x
            if i > 0:
                left_limit = max(left_limit, (raw_spans[i - 1][1] + raw_x) // 2)
            right_limit = word.bbox.x + word.bbox.w
            if i < len(matches) - 1:
                right_limit = min(right_limit, (raw_end + raw_spans[i + 1][0]) // 2)

            sub_x = max(left_limit, raw_x - _SPLIT_TOKEN_PAD_PX)
            sub_x2 = min(right_limit, raw_end + _SPLIT_TOKEN_PAD_PX)
            sub_w = max(1, sub_x2 - sub_x)
            result.append(
                EnsembleWord(
                    text=match.group(),
                    bbox=BBox(x=sub_x, y=word.bbox.y, w=sub_w, h=word.bbox.h),
                    ocr_confidence=word.ocr_confidence,
                    engine_agreement=word.engine_agreement,
                    engines=word.engines,
                    page=word.page,
                    char_start=word.char_start + match.start(),
                    char_end=word.char_start + match.end(),
                )
            )
    return result


def _group_rows(words: list[EnsembleWord]) -> list[list[EnsembleWord]]:
    """Cluster words into visual rows by vertical-center adjacency.

    Sorts by vertical center and starts a new row whenever the gap to the
    *previous* word's center exceeds tolerance. An earlier version instead
    grew an accumulating ``[top, bottom]`` envelope and tested each new
    word against it — on a real densely-packed table (row height roughly
    equal to the row-to-row gap, as scanner-grade documents commonly are)
    that envelope keeps re-expanding and chain-merges the header row plus
    every data row into a single giant "row", which then fails the header
    row-length guard entirely. Comparing only to the immediately preceding
    center avoids that runaway growth.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.bbox.y + w.bbox.h / 2, w.bbox.x))
    rows: list[list[EnsembleWord]] = []
    current = [ordered[0]]
    last_mid = ordered[0].bbox.y + ordered[0].bbox.h / 2
    for word in ordered[1:]:
        mid = word.bbox.y + word.bbox.h / 2
        tolerance = max(_ROW_Y_TOLERANCE_FLOOR, word.bbox.h * _ROW_Y_TOLERANCE_FACTOR)
        if mid - last_mid <= tolerance:
            current.append(word)
        else:
            current.sort(key=lambda w: w.bbox.x)
            rows.append(current)
            current = [word]
        last_mid = mid
    current.sort(key=lambda w: w.bbox.x)
    rows.append(current)
    return rows


def _reading_order(words: list[EnsembleWord]) -> list[EnsembleWord]:
    """Sort words into natural top-to-bottom, left-to-right reading order.

    A flat sort by x alone breaks down once ``words`` spans more than one
    visual line whose x-ranges overlap — e.g. a table-cell value wrapped
    onto a second, left-aligned line sits at nearly the same x positions
    as the first line, so an x-only sort interleaves the two lines' words
    instead of preserving line order (observed directly: a wrapped fund
    name read back as "Selaras Blife Link Plus Campuran" instead of
    "Blife Link Campuran Selaras Plus"). Reuses ``_group_rows``'s own
    y-adjacency clustering — already tuned to keep one real line's OCR
    y-jitter together while splitting genuinely distinct lines — to
    bucket into lines first, then concatenates those buckets top-to-bottom.
    """
    return [w for line in _group_rows(words) for w in line]


_CELL_OVERLAP_THRESHOLD = 0.6


def _word_overlap_fraction(word_bbox: BBox, cell_bbox: BBox) -> float:
    """Fraction of ``word_bbox``'s own area covered by ``cell_bbox``.

    Deliberately word-area-relative rather than IoU or center-containment:
    on a real scanned/photographed table, adjacent cells' Docling bboxes
    commonly overlap each other by a couple of pixels at their shared edge
    (row/column span detection isn't pixel-perfect), so a word sitting
    right at that boundary can have its *center* land inside the wrong
    neighboring cell — which, downstream, would wrongly chain-merge two
    unrelated table rows together. Requiring most of the word's own area
    (not just its center point, and not a symmetric IoU that also
    penalizes a small word against a much larger cell) to be covered
    correctly favors whichever cell the word actually, substantially
    belongs to.
    """
    ax2, ay2 = word_bbox.x + word_bbox.w, word_bbox.y + word_bbox.h
    bx2, by2 = cell_bbox.x + cell_bbox.w, cell_bbox.y + cell_bbox.h
    ix1, iy1 = max(word_bbox.x, cell_bbox.x), max(word_bbox.y, cell_bbox.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = word_bbox.w * word_bbox.h
    return inter / area if area > 0 else 0.0


def _merge_oversegmented_cells(cells: list[DocBlock]) -> list[DocBlock]:
    """Merge vertically-adjacent sub-cells in the same column into logical cells.

    Docling often reports a single multi-line table cell as one sub-cell
    per text line (observed directly on the user's table: the bank-name
    cell "Standard Chartered Bank" was split into two cells, and wrapped
    account-name cells were split similarly).  When those sub-cells are
    kept separate, the row-stitching logic sees them as distinct cells and
    cannot reunite the lines into one candidate.  This merges sub-cells
    that are vertically close and aligned in the same column, while
    leaving the larger gap between real table rows intact.

    img2table cells (``block_id`` prefixed ``i2t-``) are excluded from this
    merge and passed through unchanged: they're read directly off real
    detected gridlines, so each one already *is* the true logical cell
    boundary — the fragment-per-line failure mode this function exists to
    correct only ever comes from Docling's own TableFormer output (see
    ``table_geometry.py``'s module docstring). Re-running the vertical-gap
    merge below on img2table cells anyway is actively harmful: on a
    tightly, correctly bordered table, one real row's cell sits immediately
    (near-zero gap) above the next real row's cell in the same column —
    indistinguishable from a genuinely wrapped cell's fragmented lines by
    gap alone — which silently welds separate table rows into one,
    corrupting every row-stitching decision downstream (observed directly:
    an entire header + all data rows chain-merged into a single row-group,
    which then also fails the header row-length guard outright).
    """
    if not cells:
        return []
    cell_blocks = [c for c in cells if c.block_type == "cell" and c.bbox.w > 0 and c.bbox.h > 0]
    if not cell_blocks:
        return []

    img2table_cells = [c for c in cell_blocks if c.block_id.startswith("i2t-")]
    cell_blocks = [c for c in cell_blocks if not c.block_id.startswith("i2t-")]
    if not cell_blocks:
        return img2table_cells

    # Group cells by column using x-overlap.  Explicit table_column metadata
    # is intentionally ignored here: it can be None or noisy from img2table, and
    # for a well-spaced grid the cell x-ranges are the most reliable signal.
    sorted_by_x = sorted(cell_blocks, key=lambda c: c.bbox.x)
    columns: list[list[DocBlock]] = []
    for cell in sorted_by_x:
        cell_right = cell.bbox.x + cell.bbox.w
        placed = False
        for col in columns:
            col_left = min(c.bbox.x for c in col)
            col_right = max(c.bbox.x + c.bbox.w for c in col)
            overlap = min(col_right, cell_right) - max(col_left, cell.bbox.x)
            if overlap > 0:
                col.append(cell)
                placed = True
                break
        if not placed:
            columns.append([cell])

    merged: list[DocBlock] = []
    for col in columns:
        col.sort(key=lambda c: c.bbox.y)
        heights = sorted(c.bbox.h for c in col)
        median_h = heights[len(heights) // 2] if heights else 1.0
        max_merge_gap = max(1.0, median_h * 0.5)

        run: list[DocBlock] = [col[0]]
        for cell in col[1:]:
            prev = run[-1]
            gap = cell.bbox.y - (prev.bbox.y + prev.bbox.h)
            if gap <= max_merge_gap:
                run.append(cell)
            else:
                merged.append(_union_cell_run(run))
                run = [cell]
        if run:
            merged.append(_union_cell_run(run))

    return merged + img2table_cells


def _union_cell_run(run: list[DocBlock]) -> DocBlock:
    """Union bbox + concatenated text for a run of vertically-adjacent cells."""
    x = min(c.bbox.x for c in run)
    y = min(c.bbox.y for c in run)
    w = max(c.bbox.x + c.bbox.w for c in run) - x
    h = max(c.bbox.y + c.bbox.h for c in run) - y
    texts = [c.text for c in run if c.text]
    text = " ".join(texts) if texts else (run[0].text or "")
    return DocBlock(
        block_id=run[0].block_id,
        block_type="cell",
        bbox=BBox(x=x, y=y, w=w, h=h),
        text=text,
        table_row=run[0].table_row,
        table_column=run[0].table_column,
    )


def _assign_cell_ids(words: list[EnsembleWord], cells: list[DocBlock]) -> dict[int, int]:
    """Map each word (by ``id()``) to the Docling table-cell it substantially overlaps.

    A word is assigned to whichever cell covers the largest fraction of its
    own area, but only when that fraction clears ``_CELL_OVERLAP_THRESHOLD``
    — a weak/partial overlap (e.g. a sliver at a shared cell boundary, or
    incidental overlap with an unrelated nearby cell) isn't enough to claim
    the word, since a wrong assignment here can chain-merge unrelated table
    rows together in ``_stitch_docling_cells``. Words with no sufficiently-
    overlapping cell — not part of any table, or ``cells`` is empty — get
    no entry; callers treat a missing mapping as "not part of any cell",
    matching today's behavior exactly.
    """
    cell_ids: dict[int, int] = {}
    if not cells:
        return cell_ids
    for word in words:
        best_i, best_frac = -1, 0.0
        for i, cell in enumerate(cells):
            frac = _word_overlap_fraction(word.bbox, cell.bbox)
            if frac > best_frac:
                best_i, best_frac = i, frac
        if best_i != -1 and best_frac >= _CELL_OVERLAP_THRESHOLD:
            cell_ids[id(word)] = best_i
    return cell_ids


_ROW_CELL_OVERLAP_THRESHOLD = 0.7
# img2table's OpenCV border detection occasionally can't resolve a faint/
# broken gridline between two genuinely separate table rows and emits one
# cell spanning both of them (observed directly: a "Jumlah"/name cell twice
# the height of its neighbors). Such a cell's *vertical span* can still
# cover an ordinary, fully-populated data row's own span almost entirely
# (the row-height check above passes), even though only one or two of that
# row's many words actually fell inside it. A genuine wrapped-cell
# continuation fragment, by contrast, has *most* of its words inside that
# one cell (per the docstring: "its only content is that cell's text").
# Requiring this in addition to the row-height check is what tells the two
# cases apart, since a spurious double-height cell otherwise looks
# identical to a real multi-line wrap from the row-span check alone.
_ROW_CELL_WORD_FRACTION_THRESHOLD = 0.6


def _stitch_docling_cells(
    rows: list[list[EnsembleWord]], cell_ids: dict[int, int], cells: list[DocBlock]
) -> list[list[EnsembleWord]]:
    """Merge row-groups that share a common Docling table-cell id.

    ``_group_rows`` clusters purely by vertical adjacency, which can split a
    multi-line-wrapped cell (e.g. a 3-line bank name) into several row
    groups whenever a neighboring column's cell is shorter. Docling's own
    table-structure model recognizes the wrapped cell as a single cell
    spanning all its lines; when two or more row groups each contain a word
    mapped to that same cell, they're really one physical table row and get
    unioned back together here.

    A single word's cell mapping (``_assign_cell_ids``) isn't trusted on
    its own to justify a merge: the dual-OCR-engine ensemble commonly
    emits one duplicate/stray reading of a word at a noticeably wrong
    position (see this table's real duplicate ``"Custody"`` readings 80px
    apart), and that lone mispositioned word can still substantially
    overlap a *neighboring* row's cell even after the per-word overlap
    check. A genuinely wrapped cell's own row, by contrast, is almost
    entirely covered by that cell vertically (its only content *is* that
    cell's text). So a row is only accepted as belonging to a cell when
    the row's own vertical span is at least ``_ROW_CELL_OVERLAP_THRESHOLD``
    covered by the cell's bbox — filtering out a single stray word's
    cross-row noise while still recognizing a real one-line-per-row
    fragment (whose entire span *is* that one word) — *and* when at least
    ``_ROW_CELL_WORD_FRACTION_THRESHOLD`` of the row's own words are
    themselves mapped to that cell, which is what actually distinguishes a
    genuine wrapped-cell fragment from an ordinary, fully-populated data row
    that merely happens to fall inside a spuriously double-height img2table
    cell (a real, observed border-detection failure on tightly-packed
    tables) — without this second check, that single misdetected cell can
    transitively chain-merge every data row in the table into one, wiping
    out the whole table's redaction candidates.

    Only ever *joins* existing row groups — it can never split one — so a
    wrong link at worst over-merges (extra unrelated words land in a row,
    which the column-zone/fuzzy-match logic downstream will simply fail to
    match) rather than re-introducing new fragmentation.

    Rows with no cell-mapped words (not inside any detected table, or
    ``cell_ids`` is empty) pass through unchanged, in their original
    relative order.
    """
    if not cell_ids or not rows:
        return rows

    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    cell_to_row: dict[int, int] = {}
    for row_idx, row in enumerate(rows):
        row_top, row_bottom = _row_span(row)
        row_height = row_bottom - row_top
        if row_height <= 0:
            continue
        row_cell_id_counts = Counter(cell_ids[id(w)] for w in row if id(w) in cell_ids)
        for cell_id, word_count in row_cell_id_counts.items():
            if word_count / len(row) < _ROW_CELL_WORD_FRACTION_THRESHOLD:
                continue
            cell_bbox = cells[cell_id].bbox
            covered = min(row_bottom, cell_bbox.y + cell_bbox.h) - max(row_top, cell_bbox.y)
            if covered / row_height < _ROW_CELL_OVERLAP_THRESHOLD:
                continue
            first_row = cell_to_row.setdefault(cell_id, row_idx)
            if first_row != row_idx:
                union(first_row, row_idx)

    groups: dict[int, list[EnsembleWord]] = {}
    for row_idx, row in enumerate(rows):
        groups.setdefault(find(row_idx), []).extend(row)

    merged = list(groups.values())
    merged.sort(key=lambda row_words: min(w.bbox.y for w in row_words))
    for i, row_words in enumerate(merged):
        merged[i] = _reading_order(row_words)
    return merged


def _label_window_widths(labels: tuple[str, ...], max_width: int) -> list[int]:
    """Distinct token counts among the label phrases, capped at max_width.

    Only trying widths that match an actual phrase length (rather than a
    blind 1..max_width range) avoids matching a padded window — e.g. a
    decoy ``"No Jumlah Nama Rekening"`` (4 words, 2 of them noise) can
    still clear a similarity threshold against the 2-word ``"nama
    rekening"`` if wider windows are allowed, silently absorbing
    neighboring unrelated columns into the match.
    """
    widths = {len(label.split()) for label in labels}
    return sorted(w for w in widths if 1 <= w <= max_width)


_MIN_TOKEN_ALIGN = 0.55


def _all_words_align_to_label(window_words: list[EnsembleWord], label: str) -> bool:
    """True if every word in ``window_words`` individually resembles at
    least one word of ``label`` — see the caller in
    ``_find_all_label_windows_multi`` for why this guard exists.
    """
    label_tokens = _normalize_label_candidate(label).split()
    if not label_tokens:
        return False
    for w in window_words:
        token = _normalize_label_candidate(w.text)
        if not token:
            continue
        if max(token_sort_ratio(token, lt) for lt in label_tokens) < _MIN_TOKEN_ALIGN:
            return False
    return True


def _find_all_label_windows_multi(
    row: list[EnsembleWord],
    groups: tuple[tuple[tuple[str, ...], str], ...],
    threshold: float,
    max_width: int = _MAX_LABEL_WINDOW,
) -> list[tuple[int, int, float, str]]:
    """Non-overlapping ``(start, end, ratio, group_key)`` fuzzy matches in a row.

    All label groups are compared together at each starting position so the
    globally best-scoring group wins any overlap. Checking one group's
    phrases to completion before ever trying another's would let a partial
    match claim the tokens first — e.g. ``"account no"`` partially resembles
    ``"account name"`` (shared word ``"account"``) but must always lose to
    its own exact match in the account-number group.

    Every token in a candidate window must contribute at least one
    alphanumeric character. Without this, a stray punctuation-only token
    (e.g. a stray ``"|"`` glyph) silently degrades a two-word window like
    ``"| Rekening"`` into a one-real-word comparison against a two-word
    label (``token_sort_ratio`` tokenizes away the punctuation), which can
    score deceptively high against ``"no rekening"`` purely from the shared
    word ``"rekening"`` — a common component of every label in this table's
    header — even though the window never actually contained ``"no"``.

    A multi-word window is also rejected if one of its own tokens is
    already a strong standalone match for a *different* group's single-word
    label (e.g. ``"bank"``). Without this, a stray/garbled ``"Rekening"``
    token immediately before a real ``"Bank"`` header cell forms a two-word
    window — ``"Rekening Bank"`` — that (via ``token_sort_ratio``'s
    order-independence) still scores a deceptive ~0.83 against ``"no
    rekening"``, greedily consuming the row's real ``"Bank"`` token before
    it ever gets a chance to match on its own. Raising the overall
    threshold to reject that ~0.83 isn't a safe fix: real scanned headers
    also produce single-word OCR misreads of "Bank" itself (e.g. "Bark")
    that only clear ~0.75, so a blanket higher bar throws those out too.
    """
    single_word_labels = [
        (label, key) for labels, key in groups for label in labels if len(label.split()) == 1
    ]
    n = len(row)
    results: list[tuple[int, int, float, str]] = []
    i = 0
    while i < n:
        best: tuple[int, float, str] | None = None  # width, ratio, group_key
        for labels, key in groups:
            widths = _label_window_widths(labels, max_width)
            for width in widths:
                if i + width > n:
                    continue
                window_words = row[i : i + width]
                if any(not any(ch.isalnum() for ch in w.text) for w in window_words):
                    continue
                if width > 1 and any(
                    other_key != key
                    and token_sort_ratio(_normalize_label_candidate(w.text), other_label)
                    >= _STOLEN_TOKEN_THRESHOLD
                    for w in window_words
                    for other_label, other_key in single_word_labels
                ):
                    continue
                window_text = _normalize_label_candidate(
                    " ".join(w.text for w in window_words)
                )
                if not window_text:
                    continue
                match = best_fuzzy_match(window_text, list(labels), threshold=threshold)
                if match is not None:
                    label_idx, ratio = match
                    if width > 1 and not _all_words_align_to_label(
                        window_words, labels[label_idx]
                    ):
                        # token_sort_ratio compares the whole window against
                        # the whole label at once, so a window straddling two
                        # unrelated concepts (e.g. the tail of a wrapped
                        # address line "...INDIA" immediately followed by an
                        # unrelated field's own label start "Account") can
                        # still score high purely off *one* shared word
                        # ("account") while the other word ("india") has no
                        # real resemblance to anything in the label ("account
                        # name") — silently dragging unrelated leading text
                        # into the match. Require every window word to
                        # individually resemble some word of the label it
                        # supposedly matched.
                        continue
                    if best is None or ratio > best[1]:
                        best = (width, ratio, key)
        if best is not None:
            width, ratio, key = best
            results.append((i, i + width, ratio, key))
            i += width
        else:
            i += 1
    return results


def _find_all_label_windows(
    row: list[EnsembleWord],
    labels: tuple[str, ...],
    threshold: float,
    max_width: int = _MAX_LABEL_WINDOW,
) -> list[tuple[int, int, float]]:
    """Non-overlapping ``(start, end, ratio)`` fuzzy label matches in a row."""
    return [
        (start, end, ratio)
        for start, end, ratio, _key in _find_all_label_windows_multi(
            row, ((labels, "_"),), threshold, max_width
        )
    ]


def _find_label_in_row(
    row: list[EnsembleWord],
    labels: tuple[str, ...],
    threshold: float = 0.72,
    require_start_within: int = 3,
) -> tuple[int, int, float] | None:
    """Best label match that starts within the first ``require_start_within`` words."""
    for start, end, ratio in _find_all_label_windows(row, labels, threshold=threshold):
        if start <= require_start_within:
            return start, end, ratio
    return None


def _find_best_group_match(
    row: list[EnsembleWord],
    groups: tuple[tuple[tuple[str, ...], str], ...],
    threshold: float = 0.72,
    require_start_within: int = 3,
) -> tuple[int, int, float, str] | None:
    """Best cross-group label match that starts within the row's first few words."""
    for start, end, ratio, key in _find_all_label_windows_multi(row, groups, threshold=threshold):
        if start <= require_start_within:
            return start, end, ratio, key
    return None


def _value_after(row: list[EnsembleWord], end_idx: int) -> list[EnsembleWord]:
    """Words following a label match, skipping a leading colon/dash token."""
    tail = row[end_idx:]
    while tail and tail[0].text.strip(" :.-") == "":
        tail = tail[1:]
    return tail


# --- Table-column mode ------------------------------------------------


@dataclass
class _HeaderAnchor:
    x_center: float
    base_role: str
    left: float
    right: float


@dataclass
class _ColumnDef:
    x_center: float
    role: field_labels.FieldRole
    number_x_center: float | None = None


_HEADER_LABEL_GROUPS: tuple[tuple[tuple[str, ...], str], ...] = (
    (field_labels.ACCOUNT_NAME_LABELS, _BASE_ROLE_ACCOUNT_NAME),
    (field_labels.BANK_LABELS, _BASE_ROLE_BANK),
    (field_labels.ACCOUNT_NUMBER_LABELS, _BASE_ROLE_ACCOUNT_NUMBER),
)


def _match_header_row(row: list[EnsembleWord]) -> list[_HeaderAnchor]:
    """Find non-overlapping account-name / bank / account-number header hits."""
    anchors: list[_HeaderAnchor] = []
    for start, end, _ratio, base_role in _find_all_label_windows_multi(
        row, _HEADER_LABEL_GROUPS, threshold=0.72
    ):
        words = row[start:end]
        x_center = sum(w.bbox.x + w.bbox.w / 2 for w in words) / len(words)
        left = min(w.bbox.x for w in words)
        right = max(w.bbox.x + w.bbox.w for w in words)
        anchors.append(_HeaderAnchor(x_center=x_center, base_role=base_role, left=left, right=right))
    return anchors


_MIN_HEADER_ANCHORS = 2


def _detect_table_header(
    rows: list[list[EnsembleWord]],
) -> tuple[int, list[_HeaderAnchor]] | None:
    """Row with the most recognized column headers.

    Requires >=1 account-name hit *and* >=2 total header hits in the row —
    a genuine multi-column table header names several columns at once
    (``Nama Rekening`` + ``Bank`` + ``No Rekening``, twice over for
    debit/credit). A single isolated ``A/C Name`` match is far more likely
    to be an ordinary label:value content row (handled by
    ``_label_value_candidates`` instead), so it's rejected here.
    """
    best_idx = -1
    best_anchors: list[_HeaderAnchor] = []
    for idx, row in enumerate(rows):
        if len(row) > _HEADER_MAX_ROW_WORDS:
            continue
        anchors = _match_header_row(row)
        name_count = sum(1 for a in anchors if a.base_role == _BASE_ROLE_ACCOUNT_NAME)
        if (
            name_count >= 1
            and len(anchors) >= _MIN_HEADER_ANCHORS
            and len(anchors) > len(best_anchors)
        ):
            best_idx = idx
            best_anchors = anchors
    if best_idx == -1:
        return None
    return best_idx, sorted(best_anchors, key=lambda a: a.x_center)


def _build_column_defs(anchors: list[_HeaderAnchor]) -> list[_ColumnDef]:
    """Map header anchors to redaction roles; pair number columns to name columns.

    A number column is paired to the name column that precedes it (by x)
    with no other name column in between — i.e. the account-number column
    for "belongs to" whichever name column starts its group, reading
    left-to-right (name, bank, number, name, bank, number, ...). Nearest-
    by-absolute-distance would misassign this when columns are evenly
    spaced, since a debit-side number column can sit numerically closer to
    the credit-side name column than to its own.
    """
    name_anchors = sorted(
        (a for a in anchors if a.base_role == _BASE_ROLE_ACCOUNT_NAME),
        key=lambda a: a.x_center,
    )
    bank_anchors = [a for a in anchors if a.base_role == _BASE_ROLE_BANK]
    number_anchors = [a for a in anchors if a.base_role == _BASE_ROLE_ACCOUNT_NUMBER]
    name_roles: list[field_labels.FieldRole] = ["debit_account_name", "credit_account_name"]

    columns: list[_ColumnDef] = []
    for i, anchor in enumerate(name_anchors):
        role = name_roles[i] if i < len(name_roles) else "counterparty_org"
        zone_end = name_anchors[i + 1].x_center if i + 1 < len(name_anchors) else float("inf")
        in_zone = [
            n.x_center for n in number_anchors if anchor.x_center <= n.x_center < zone_end
        ]
        number_x = min(in_zone) if in_zone else None
        columns.append(_ColumnDef(x_center=anchor.x_center, role=role, number_x_center=number_x))
    for anchor in bank_anchors:
        columns.append(_ColumnDef(x_center=anchor.x_center, role="bank_name"))
    return columns


def _max_bucket_distance(anchors: list[_HeaderAnchor]) -> float:
    """Half the smallest gap between adjacent header columns, clamped to a sane range."""
    xs = sorted(a.x_center for a in anchors)
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 0]
    if not gaps:
        return 150.0
    return max(15.0, min(250.0, min(gaps) / 2))


def _nearest_outside_edge(
    outside_words: list[EnsembleWord], *, before: float | None, after: float | None
) -> float | None:
    """Nearest edge (right edge if ``before`` given, left edge if ``after``) of an
    unclaimed header word lying strictly on that side, or ``None`` if there is none.
    """
    if before is not None:
        candidates = [w.bbox.x + w.bbox.w for w in outside_words if w.bbox.x + w.bbox.w <= before]
        return max(candidates) if candidates else None
    candidates = [w.bbox.x for w in outside_words if w.bbox.x >= after]
    return min(candidates) if candidates else None


def _column_zones(
    anchors: list[_HeaderAnchor],
    edge_distance: float,
    header_row: list[EnsembleWord] | None = None,
) -> list[tuple[float, float, _HeaderAnchor]]:
    """Non-overlapping ``(left, right, anchor)`` x-ranges, one per header anchor.

    Interior boundaries sit at the midpoint between neighboring anchors, so
    a long cell value (e.g. a fund name wider than the column header) is
    still claimed by its own column rather than dropped for falling outside
    a fixed radius.

    The two outermost zones are unbounded on their open side by definition,
    but an unlabeled neighboring column (e.g. an ``IDR``/``USD`` amount
    column to the left of the first name column) sits much closer to the
    anchor than half the gap *between recognized headers* would suggest —
    that gap is measured against the *next* recognized header, which can be
    a whole other column away. So when ``header_row`` is given, the outer
    boundary snaps to the midpoint against the nearest other header-row
    token (recognized or not) on that side, matching where the real column
    divider sits; falling back to the old fixed ``edge_distance`` radius
    only when there's no such neighboring token (e.g. table starts at the
    page edge).
    """
    ordered = sorted(anchors, key=lambda a: a.x_center)
    outside_words: list[EnsembleWord] = []
    if header_row is not None:
        claimed = [(a.left, a.right) for a in ordered]
        outside_words = [
            w
            for w in header_row
            if not any(lo <= w.bbox.x and w.bbox.x + w.bbox.w <= hi for lo, hi in claimed)
        ]
    zones: list[tuple[float, float, _HeaderAnchor]] = []
    for i, anchor in enumerate(ordered):
        if i == 0:
            neighbor_edge = _nearest_outside_edge(outside_words, before=anchor.left, after=None)
            left = (
                (neighbor_edge + anchor.left) / 2
                if neighbor_edge is not None
                else anchor.x_center - edge_distance
            )
        else:
            left = (ordered[i - 1].x_center + anchor.x_center) / 2
        if i == len(ordered) - 1:
            neighbor_edge = _nearest_outside_edge(outside_words, before=None, after=anchor.right)
            right = (
                (anchor.right + neighbor_edge) / 2
                if neighbor_edge is not None
                else anchor.x_center + edge_distance
            )
        else:
            right = (anchor.x_center + ordered[i + 1].x_center) / 2
        zones.append((left, right, anchor))
    return zones


_ZONE_WORD_OVERLAP_THRESHOLD = 0.7


def _zone_words(row: list[EnsembleWord], left: float, right: float) -> list[EnsembleWord]:
    """Row words substantially inside ``(left, right]``, left-to-right.

    Requires most (``_ZONE_WORD_OVERLAP_THRESHOLD``) of a word's own width
    to fall inside the zone, not just its center point. OCR occasionally
    glues a value from the *previous* column onto the first word of this
    one into a single token — no space between them in the scan, e.g. an
    amount like "0.00" fused with the following "Blife" into
    "O.00@life" — and such a token's bbox straddles the real column
    boundary. Its center can land just barely inside this zone even
    though roughly half its own width truly belongs to the neighboring
    column, which a pure center check would let leak straight through
    (including the neighbor's numeric prefix). Requiring substantial
    width overlap excludes that glued case while still tolerating the
    ordinary few-pixel jitter of a word that's genuinely part of this
    column.
    """
    picked = []
    for w in row:
        if w.bbox.w <= 0:
            continue
        overlap = min(w.bbox.x + w.bbox.w, right) - max(w.bbox.x, left)
        if overlap / w.bbox.w >= _ZONE_WORD_OVERLAP_THRESHOLD:
            picked.append(w)
    return _reading_order(picked)


_PREFIX_BOUNDARY_TOLERANCE_PX = 6


def _rescue_left_prefix_words(
    row: list[EnsembleWord], base: dict[float, list[EnsembleWord]]
) -> None:
    """Pull a known org-name prefix (e.g. "PT", "CV") into the zone whose
    leftmost accepted word sits immediately to its right, in place.

    A column boundary here is only the midpoint between two header labels'
    x-centers, not the real grid line — a fair estimate for ordinary
    words, but a short 2-3 letter prefix straddling that estimate can have
    most of its own width land on the wrong side from just a few pixels of
    misalignment (e.g. the header column's numbers are wider than its own
    "No Rekening" label, shifting where the *real* boundary sits). That
    same misalignment barely dents a normal-length word's overlap
    fraction, so ``_zone_words``'s width-overlap test disproportionately
    drops short prefixes right at a zone's left edge. Since a leading org
    prefix is a closed, well-known vocabulary
    (``field_labels.ORG_PREFIX_STOPWORDS``) and is never a plausible
    "glued-on amount from the previous column" (it's alphabetic, not
    numeric), it's safe to special-case it in in place of a general
    boundary-tolerance relaxation that would risk leaking real neighboring
    text.
    """
    claimed_ids = {id(w) for words in base.values() for w in words}
    for words in base.values():
        if not words:
            continue
        first = words[0]
        best: EnsembleWord | None = None
        for w in row:
            if id(w) in claimed_ids:
                continue
            text = w.text.strip(" .").lower()
            if text not in field_labels.ORG_PREFIX_STOPWORDS:
                continue
            gap = first.bbox.x - (w.bbox.x + w.bbox.w)
            if gap < -_PREFIX_BOUNDARY_TOLERANCE_PX or gap > w.bbox.w:
                continue
            y_overlap = min(w.bbox.y + w.bbox.h, first.bbox.y + first.bbox.h) - max(
                w.bbox.y, first.bbox.y
            )
            if y_overlap <= 0.5 * min(w.bbox.h, first.bbox.h):
                continue
            if best is None or w.bbox.x > best.bbox.x:
                best = w
        if best is not None:
            words.insert(0, best)
            claimed_ids.add(id(best))


def _zone_words_by_anchor(
    row: list[EnsembleWord],
    zones: list[tuple[float, float, _HeaderAnchor]],
    cell_ids: dict[int, int],
) -> dict[float, list[EnsembleWord]]:
    """Per-anchor zone words (keyed by ``x_center``), extended to each word's
    full Docling table cell.

    ``_zone_words``'s x-range test is a header-anchor-derived *estimate* of
    where a column sits — it can under-capture a cell whose real content is
    simply wider than the header label it was measured from (e.g. a long
    fund name like "... Saham Maksima Plus" overflowing the narrow zone
    implied by the short "Nama Rekening" header text, dropping "Maksima"
    off the end and leaving the remaining truncated text to fuzzy-match a
    *different*, shorter curated name instead of its own). When Docling
    already identified several of a zone's words as sharing one physical
    table cell, any other word in that same cell is part of the same
    column value even if its raw x position falls just outside the
    estimated zone — a stronger, structural signal than the x-range guess.
    Each cell is only ever folded into the single zone that first claimed
    one of its words, so a word is never double-counted across zones.

    A candidate word is only pulled in if it's safely distinguishable from
    that "wrong neighboring column merged in" failure mode, which always
    shows up as extra text on the *same OCR line* as (i.e. vertically
    overlapping) an already-accepted word but positioned to its *left* —
    exactly how ``_split_multiword_tokens`` produces an amount's and a
    name's sub-tokens out of one merged "0.00 Some Fund" detection. So a
    candidate is accepted when either:
      * it doesn't vertically overlap any already-accepted word in the
        zone at all — a genuinely different physical line, e.g. one line
        of a multi-line-wrapped cell that a per-word overlap fraction
        (``_zone_words``) narrowly missed; or
      * it does overlap a same-line accepted word, but only as a trailing
        continuation strictly to that word's right — e.g. a long value's
        last word overflowing the zone's right edge.
    Each cell is only ever folded into the single zone that first claimed
    one of its words, so a word is never double-counted across zones.
    Note this deliberately still blocks a *legitimate* leading prefix
    (e.g. org name's "PT") that happens to share a Docling cell with the
    rest of its value — ``_rescue_left_prefix_words`` (run beforehand,
    independent of Docling cells) is what recovers those specifically, by
    checking against a closed prefix vocabulary rather than loosening this
    generic, garbled-OCR-prone position check.

    The x-range test can also *over*-capture: a wide column's header
    label sits well inside the column, so the header-anchor midpoint used
    as this zone's boundary can fall short of the column's true right
    edge — a wrapped cell's later, longer lines then drift past that
    boundary and get pulled directly into the *neighboring* zone's own
    raw x-range instead (the mirror image of the under-capture case
    above: this word matches some zone's test, just the wrong one). A
    word whose Docling cell is shared by several other words already
    settled in a different zone is reassigned to that zone — the cell's
    majority vote is trusted over any single word's raw x-position.
    """
    base = {anchor.x_center: _zone_words(row, left, right) for left, right, anchor in zones}
    _rescue_left_prefix_words(row, base)
    if not cell_ids:
        return base
    cell_zone_votes: dict[int, Counter[float]] = {}
    for x_center, words in base.items():
        for w in words:
            cell_id = cell_ids.get(id(w))
            if cell_id is not None:
                cell_zone_votes.setdefault(cell_id, Counter())[x_center] += 1
    dominant_zone_for_cell = {
        cell_id: votes.most_common(1)[0][0] for cell_id, votes in cell_zone_votes.items()
    }
    for x_center, words in base.items():
        keep, evict = [], []
        for w in words:
            cell_id = cell_ids.get(id(w))
            dominant = dominant_zone_for_cell.get(cell_id) if cell_id is not None else None
            (evict if dominant is not None and dominant != x_center else keep).append(w)
        base[x_center] = keep
        for w in evict:
            base[dominant_zone_for_cell[cell_ids[id(w)]]].append(w)

    claimed_ids = {id(w) for words in base.values() for w in words}
    zone_for_cell: dict[int, float] = dict(dominant_zone_for_cell)

    def _shares_line_to_the_left(
        candidate: EnsembleWord, accepted: list[EnsembleWord], own_cell_id: int
    ) -> bool:
        for a in accepted:
            if cell_ids.get(id(a)) == own_cell_id:
                # Same physical Docling/img2table cell as the candidate —
                # this is the candidate's own line-mate within one wrapped
                # value (e.g. the first word of a wrap's 2nd/3rd line,
                # which legitimately starts to the left of that line's
                # other, already-raw-accepted words), not a different
                # column's text glued onto this line by OCR.
                continue
            y_overlap = min(candidate.bbox.y + candidate.bbox.h, a.bbox.y + a.bbox.h) - max(
                candidate.bbox.y, a.bbox.y
            )
            same_line = y_overlap > 0.5 * min(candidate.bbox.h, a.bbox.h)
            if same_line and candidate.bbox.x <= a.bbox.x:
                return True
        return False

    for w in sorted(row, key=lambda w: w.bbox.x):
        if id(w) in claimed_ids:
            continue
        cell_id = cell_ids.get(id(w))
        if cell_id is None:
            continue
        x_center = zone_for_cell.get(cell_id)
        if x_center is None or _shares_line_to_the_left(w, base[x_center], cell_id):
            continue
        base[x_center].append(w)
        claimed_ids.add(id(w))
    for x_center in list(base):
        base[x_center] = _reading_order(base[x_center])
    return base


_TABLE_END_GAP_MULTIPLIER = 4.0


def _row_span(row: list[EnsembleWord]) -> tuple[float, float]:
    return min(w.bbox.y for w in row), max(w.bbox.y + w.bbox.h for w in row)


def _row_zone_membership(
    row: list[EnsembleWord],
    zones: list[tuple[float, float, _HeaderAnchor]],
    cell_ids: dict[int, int],
) -> set[float]:
    """Which zones' x-centers this row has substantial word overlap with.

    Uses ``_zone_words_by_anchor`` (cell-rescue-aware) rather than the raw
    ``_zone_words`` x-range test: a wide column whose header label is much
    narrower than its actual content (e.g. "Account Name" labeling a wide
    fund-name column) has its zone boundary underestimated from the
    header-anchor midpoint alone, so a wrapped continuation line's later
    words can fall just outside the estimated zone and get miscounted
    against a neighboring column's zone instead. That false extra
    membership makes ``_column_band_stitch`` treat a genuine single-role
    fragment as a multi-role anchor and skip merging it. Docling's cell
    geometry (``cell_ids``) resolves this the same way it already does for
    the final candidate-building pass.
    """
    zone_words = _zone_words_by_anchor(row, zones, cell_ids)
    return {x_center for x_center, words in zone_words.items() if words}


def _column_band_stitch(
    rows: list[list[EnsembleWord]],
    header: tuple[int, list[_HeaderAnchor]] | None,
    cell_ids: dict[int, int] | None = None,
) -> list[list[EnsembleWord]]:
    """OCR-geometry-only fallback: merge a wrapped cell's continuation line
    back into its own row when Docling cell geometry didn't already do
    this (missing/low-confidence table cells — common on noisy/
    photographed scans).

    ``_stitch_docling_cells`` is the primary path and already handles this
    correctly whenever Docling identifies the wrapped cell; when it can't
    (no ``cell`` blocks at all, or the wrapped cell's lines didn't clear
    ``_ROW_CELL_OVERLAP_THRESHOLD``), a multi-line value like "Standard /
    Chartered / Custody" is left split across several row-groups by
    ``_group_rows``'s tight, uniform y-tolerance. Each of those trailing
    lines contains words from *only one* recognized table column (the
    other columns in that same logical row have just one line, so they
    contribute nothing at that y-position) — this "pure single-column
    row" shape is exactly what distinguishes a wrapped-cell continuation
    from a genuine new table row, which always has content in more than
    one column at once (name + bank + number together).

    A row's column-*band* membership (raw x-center) isn't quite the right
    granularity for deciding whether it's a fragment, though: a debit-
    side and credit-side column pair (e.g. two "Bank" columns) share the
    same ``base_role`` but sit at different x-centers, and when both
    sides wrap in lockstep — common, since debit/credit rows mirror each
    other — a single fragment row-group picks up one word from *each*
    side's band at once. Grouping bands by ``base_role`` first (a row is
    a "fragment" candidate when its bands reduce to exactly one role,
    however many bands that spans) still correctly rejects a genuine new
    table row, which always populates more than one *role* at once (name
    + bank + number together) — while no longer rejecting this mirrored-
    wrap case just because it happens to touch two bands.

    A fragment's true home row (the "anchor": the one line, among a
    wrapped cell's several, that lines up with this row's *other*
    columns) is not necessarily the nearest preceding anchor — a
    fragment can equally be a wrapped cell's *first* line, sitting above
    an anchor that hasn't been seen yet, or be the *only* wrapped line
    when the anchor's own row has no baseline occurrence of that column
    at all (nothing to share membership with). So rather than a single
    forward sweep folding into "the most recent row with this band
    open", runs of consecutive fragment rows are first delimited by the
    (at most two) anchor rows bracketing them, and each fragment attaches
    to whichever bracketing anchor's row-center it sits closer to —
    falling back to unmerged if neither bracketing anchor is within
    ``_COLUMN_BAND_Y_TOLERANCE_FACTOR`` of it. Fragments are never merged
    directly into each other (only into a bracketing anchor), since two
    independent wrapped cells' stray lines can otherwise sit closer to
    each other than to their own anchors.

    Nearest-by-y-center is itself only a proxy for "which anchor is this
    the same physical cell as" — one that fails on real scanned/
    photographed tables whenever a wrapped cell's account-number/amount
    row happens to be its *second* line rather than its first: that
    second line's own leading (the gap back up to its own name's first
    line, one row above) can be wider than the *previous* row's trailing
    gap down to this fragment, making the fragment nearest, by pure
    y-distance, to the wrong row entirely. When Docling/img2table's own
    cell geometry (``cell_ids``) puts the fragment in the exact same
    physical cell as one bracketing anchor and not the other, that
    structural signal overrides the y-distance guess (it is never
    available for a genuinely new, unrelated row, so this can't wrongly
    absorb one); ties (both sides sharing a cell — never seen in
    practice, but not a signal either way) fall back to the y-distance
    result unchanged.

    Purely additive: a row with content in zero or more-than-one role is
    never touched, so a document Docling already stitched correctly sees
    no leftover pure-single-role fragments here and this is a no-op.
    Requires a detected table ``header`` (``None`` short-circuits to a
    no-op) since column bands only exist once column positions are
    known.
    """
    if header is None or len(rows) < 2:
        return rows
    cell_ids = cell_ids or {}
    header_idx, anchors = header
    zones = _column_zones(anchors, _max_bucket_distance(anchors), rows[header_idx])
    if not zones:
        return rows

    data_rows = rows[header_idx + 1 :]
    n = len(data_rows)
    is_anchor = [True] * n
    for i, row in enumerate(data_rows):
        if not row:
            continue
        membership = _row_zone_membership(row, zones, cell_ids)
        if not membership:
            continue
        roles = {anchor.base_role for _, _, anchor in zones if anchor.x_center in membership}
        is_anchor[i] = len(roles) != 1

    merged: list[list[EnsembleWord] | None] = list(data_rows)
    i = 0
    while i < n:
        if is_anchor[i] or not data_rows[i]:
            i += 1
            continue
        run_start = i
        while i < n and not is_anchor[i] and data_rows[i]:
            i += 1
        run_end = i

        left_anchor = run_start - 1 if run_start > 0 and is_anchor[run_start - 1] else None
        right_anchor = run_end if run_end < n and is_anchor[run_end] else None

        for k in range(run_start, run_end):
            fragment = data_rows[k]
            frag_top, frag_bottom = _row_span(fragment)
            frag_center = (frag_top + frag_bottom) / 2
            tolerance = max(
                _ROW_Y_TOLERANCE_FLOOR,
                (frag_bottom - frag_top) * _COLUMN_BAND_Y_TOLERANCE_FACTOR,
            )

            best_side: int | None = None
            best_dist = None
            cell_matched_side: int | None = None
            frag_cell_ids = {cid for w in fragment if (cid := cell_ids.get(id(w))) is not None}
            for side_idx in (left_anchor, right_anchor):
                if side_idx is None:
                    continue
                anchor_row = data_rows[side_idx]
                anchor_top, anchor_bottom = _row_span(anchor_row)
                if frag_cell_ids:
                    anchor_cell_ids = {
                        cid for w in anchor_row if (cid := cell_ids.get(id(w))) is not None
                    }
                    if frag_cell_ids & anchor_cell_ids:
                        if cell_matched_side is None:
                            cell_matched_side = side_idx
                        elif cell_matched_side != side_idx:
                            cell_matched_side = -1
                gap = max(0.0, frag_top - anchor_bottom, anchor_top - frag_bottom)
                if gap > tolerance:
                    continue
                dist = abs(frag_center - (anchor_top + anchor_bottom) / 2)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_side = side_idx

            if cell_matched_side is not None and cell_matched_side != -1:
                best_side = cell_matched_side

            if best_side is not None:
                merged[best_side] = sorted(merged[best_side] + fragment, key=lambda w: w.bbox.x)
                merged[k] = None

    header_rows = list(rows[: header_idx + 1])
    data_rows_out = [row for row in merged if row]
    # A wrapped cell's first line can sit above the row's shared main
    # baseline (e.g. the credit-bank column's line 1 a few px higher than
    # the debit-side columns on the "same" logical row) — merging its
    # earlier or later lines into an anchor can leave that anchor out of
    # top-to-bottom order relative to other rows. Restore reading order
    # (mirrors _stitch_docling_cells's own re-sort) so the sequential
    # end-of-table gap heuristic in _table_column_candidates still sees
    # monotonically increasing row positions.
    data_rows_out.sort(key=lambda row_words: min(w.bbox.y for w in row_words))
    return header_rows + data_rows_out


# A fragment line and the real row it belongs to sit right against each
# other or overlap slightly (a genuine wrap continuation has ~0 gap,
# sometimes a few px of line leading); two distinct real table rows for
# the same column/role — even short ones — are always separated by a
# visible gap of at least a few percent of the page height (a real row
# boundary), well outside this margin. Deliberately tight: this must never
# fold two different rows' *legitimate, substantial* values together
# (e.g. two different fund names), only reunite a wrap line that
# ``_column_band_stitch`` failed to fold back into its own cell. Used by
# ``_merge_adjacent_same_column_candidates`` below.
_ADJACENT_MERGE_GAP_FLOOR = 4.0
_ADJACENT_MERGE_GAP_FACTOR = 0.2
# Minimum fraction of the narrower candidate's x-range that must overlap
# the other's to count as "the same physical column".
_ADJACENT_MERGE_MIN_X_OVERLAP = 0.5
# Only ever gap-merge when *both* sides are this small. A genuine stray
# wrap-line fragment left behind by ``_column_band_stitch`` is one or two
# words (its whole content is a single line of a multi-line cell), and so
# is the anchor row it's meant to rejoin before all of that row's lines
# have been reunited. Requiring *both* sides small (not just the smaller
# one) matters because consecutive real table rows can wrap onto extra
# lines back-to-back with ~0 gap between them: a fully-reunited 3-line
# value from row N (e.g. a complete "Standard Chartered Custody") can sit
# only a few px from row N+1's own first fragment line — if only the
# smaller side needed to be small, that already-complete value would
# wrongly keep absorbing the *next* row's fragments too. Once a value is
# no longer fragment-sized on either side, it's done growing.
_ADJACENT_MERGE_MAX_FRAGMENT_WORDS = 2
# Two same-column candidates whose y-ranges overlap by at least this
# fraction of the *smaller* one's own height are merged unconditionally
# (regardless of word count). Two genuinely separate real table rows
# never overlap vertically — that would mean their cells physically
# overlap on the page, which cannot happen in a real rendered table — so
# any such overlap can only mean two independently-assembled fragments
# (e.g. from different row-grouping quirks upstream) describing the same
# physical cell, which is safe to fuse regardless of how "substantial"
# each one already looks.
_ADJACENT_MERGE_MIN_Y_OVERLAP_FRACTION = 0.4


def _candidate_x_range(cand: FieldCandidate) -> tuple[float, float]:
    return min(w.bbox.x for w in cand.words), max(w.bbox.x + w.bbox.w for w in cand.words)


def _candidate_y_range(cand: FieldCandidate) -> tuple[float, float]:
    return min(w.bbox.y for w in cand.words), max(w.bbox.y + w.bbox.h for w in cand.words)


def _y_ranges_substantially_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    smaller = min(a[1] - a[0], b[1] - b[0])
    return smaller > 0 and overlap / smaller >= _ADJACENT_MERGE_MIN_Y_OVERLAP_FRACTION


def _x_ranges_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    narrower = min(a[1] - a[0], b[1] - b[0])
    return narrower > 0 and hi > lo and (hi - lo) / narrower >= _ADJACENT_MERGE_MIN_X_OVERLAP


def _merge_two_candidates(prev: FieldCandidate, cand: FieldCandidate) -> FieldCandidate:
    seen_ids = {id(w) for w in prev.words}
    combined = list(prev.words) + [w for w in cand.words if id(w) not in seen_ids]
    combined.sort(key=lambda w: (w.bbox.y, w.bbox.x))
    return FieldCandidate(
        start=min(prev.start, cand.start),
        end=max(prev.end, cand.end),
        entity_type=prev.entity_type,
        field_role=prev.field_role,
        match_confidence=max(prev.match_confidence, cand.match_confidence),
        words=tuple(combined),
        account_number=prev.account_number or cand.account_number,
    )


def _merge_adjacent_same_column_candidates(
    candidates: list[FieldCandidate],
) -> list[FieldCandidate]:
    """Fold a stray, same-column wrap-line fragment into its neighbor.

    ``_column_band_stitch``'s fallback only recognizes a fragment row as a
    wrap continuation when *all* of that row's words fall in a single
    column band. That check fails when two columns in the same physical
    table row both wrap onto an extra line at once (observed directly:
    both the debit- and credit-side "Bank" cells wrapping "Standard
    Chartered Custody" the same way put a "Standard"/"Custody" word in
    *each* column's band on the same fragment line) — the row then has
    more than one membership and is left as its own tiny, ambiguous
    "row", which this loop turns into a bogus second candidate for a
    column that already has one right above or below it.

    This runs after candidate assembly, once each candidate's own column
    is already known via ``field_role``, so it can safely merge any two
    same-role, same-column (x-overlapping) candidates that are either (a)
    a small fragment sitting right next to a neighbor, or (b) two
    candidates whose y-ranges substantially overlap outright — which can
    only happen when both describe the same physical cell, since two
    genuinely separate table rows' cells never overlap on a real
    rendered page. Either signal alone is enough; a real, separate table
    row is always a full row height away and triggers neither.

    ``field_role`` alone doesn't identify a single physical column: the
    debit- and credit-side "Bank" columns share the generic ``bank_name``
    role, so a naive single sorted-by-y sweep across a role's whole group
    would interleave the two columns' candidates and silently break
    adjacency between same-column fragments that happen to sit a few
    slots apart once the other column's candidates are mixed in. Pairs
    are checked by ``_x_ranges_overlap`` (i.e. within one column already)
    before anything else, and matches are unioned transitively — via a
    union-find over *all* pairs in the role's group, not just
    sorted-adjacent ones — so a chain of fragments merges correctly
    regardless of how the two columns' rows happen to interleave in sort
    order.
    """
    by_role: dict[str, list[FieldCandidate]] = {}
    order: list[str] = []
    for cand in candidates:
        by_role.setdefault(cand.field_role, []).append(cand)
        if cand.field_role not in order:
            order.append(cand.field_role)

    merged: list[FieldCandidate] = []
    for role in order:
        group = by_role[role]
        n = len(group)
        y_ranges = [_candidate_y_range(c) for c in group]
        x_ranges = [_candidate_x_range(c) for c in group]
        avg_heights = [sum(w.bbox.h for w in c.words) / len(c.words) for c in group]

        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for i in range(n):
            for j in range(i + 1, n):
                if not _x_ranges_overlap(x_ranges[i], x_ranges[j]):
                    continue
                both_small = (
                    max(len(group[i].words), len(group[j].words))
                    <= _ADJACENT_MERGE_MAX_FRAGMENT_WORDS
                )
                gap = max(
                    0.0,
                    y_ranges[j][0] - y_ranges[i][1],
                    y_ranges[i][0] - y_ranges[j][1],
                )
                avg_h = (avg_heights[i] + avg_heights[j]) / 2
                gap_ok = gap <= max(_ADJACENT_MERGE_GAP_FLOOR, _ADJACENT_MERGE_GAP_FACTOR * avg_h)
                fragment_adjacent = both_small and gap_ok
                overlapping = _y_ranges_substantially_overlap(y_ranges[i], y_ranges[j])
                if fragment_adjacent or overlapping:
                    union(i, j)

        components: dict[int, list[int]] = {}
        for i in range(n):
            components.setdefault(find(i), []).append(i)
        for member_idxs in components.values():
            combined = group[member_idxs[0]]
            for idx in member_idxs[1:]:
                combined = _merge_two_candidates(combined, group[idx])
            merged.append(combined)
    return merged


def _table_column_candidates(
    rows: list[list[EnsembleWord]],
    header: tuple[int, list[_HeaderAnchor]] | None,
    cell_ids: dict[int, int] | None = None,
) -> list[FieldCandidate]:
    """Detect account-name/bank table columns by header text, bucket data rows by x.

    Stops at the first row whose vertical gap from the previous row is far
    larger than the gaps seen so far — e.g. a totals line followed by a
    large whitespace gap before unrelated footer/boilerplate text below the
    table. Without this, every row all the way to the bottom of the page
    gets treated as more table data.
    """
    if header is None:
        return []
    cell_ids = cell_ids or {}
    header_idx, anchors = header
    columns = _build_column_defs(anchors)
    if not columns:
        return []
    zones = _column_zones(anchors, _max_bucket_distance(anchors), rows[header_idx])
    column_by_x = {c.x_center: c for c in columns}

    candidates: list[FieldCandidate] = []
    _, prev_bottom = _row_span(rows[header_idx])
    row_gap_reference: float | None = None
    for row in rows[header_idx + 1 :]:
        if not row:
            continue
        row_top, row_bottom = _row_span(row)
        gap = row_top - prev_bottom
        if row_gap_reference is None:
            row_gap_reference = max(gap, _ROW_Y_TOLERANCE_FLOOR)
        elif gap > row_gap_reference * _TABLE_END_GAP_MULTIPLIER:
            break
        else:
            row_gap_reference = (row_gap_reference + max(gap, 0.0)) / 2
        prev_bottom = row_bottom

        zone_words = _zone_words_by_anchor(row, zones, cell_ids)

        for x_center, words in zone_words.items():
            column = column_by_x.get(x_center)
            if column is None or not words:
                continue
            # A glued OCR token like "0.00 Blife..." can leave a numeric
            # sub-token inside the name zone after splitting; dropping it
            # keeps the painted union inside the name cell instead of
            # stretching left across the USD/amount column.
            words = [w for w in words if not _looks_numeric_or_date(w.text)]
            if not words:
                continue
            text = " ".join(w.text for w in words).strip()
            if _looks_numeric_or_date(text):
                continue
            account_number = None
            if column.number_x_center is not None:
                number_words = zone_words.get(column.number_x_center, [])
                account_number = " ".join(w.text for w in number_words).strip() or None
            candidates.append(
                FieldCandidate(
                    start=min(w.char_start for w in words),
                    end=max(w.char_end for w in words),
                    entity_type="ORGANIZATION",
                    field_role=column.role,
                    match_confidence=0.8,
                    words=tuple(words),
                    account_number=account_number,
                )
            )
    return candidates


# --- Label:value mode (letters, non-table layouts) --------------------


@dataclass
class _RowEvent:
    row_idx: int
    kind: str  # "name" | "bank" | "number" | "addressee_value"
    section: str | None
    value_words: list[EnsembleWord]
    ratio: float


def _collect_row_events(
    rows: list[list[EnsembleWord]], skip_row_idx: int | None = None
) -> list[_RowEvent]:
    events: list[_RowEvent] = []
    section: str | None = None
    pending_addressee_row: int | None = None

    for row_idx, row in enumerate(rows):
        if not row or row_idx == skip_row_idx:
            continue

        if pending_addressee_row == row_idx - 1:
            pending_addressee_row = None
            events.append(_RowEvent(row_idx, "addressee_value", section, row, 0.7))
            continue

        if len(row) <= _SECTION_MAX_ROW_WORDS:
            if _find_label_in_row(row, field_labels.DEBIT_SECTION_LABELS, threshold=0.78, require_start_within=2):
                section = "debit"
                continue
            if _find_label_in_row(row, field_labels.CREDIT_SECTION_LABELS, threshold=0.78, require_start_within=2):
                section = "credit"
                continue

        match = _find_best_group_match(row, _HEADER_LABEL_GROUPS)
        if match is not None:
            _, end_idx, ratio, kind = match
            events.append(_RowEvent(row_idx, kind, section, _value_after(row, end_idx), ratio))
            continue

        if len(row) <= _ADDRESSEE_MAX_ROW_WORDS:
            match = _find_label_in_row(row, field_labels.ADDRESSEE_LABELS, threshold=0.78, require_start_within=1)
            if match is not None:
                _, end_idx, ratio = match
                value_words = _value_after(row, end_idx)
                if value_words:
                    events.append(_RowEvent(row_idx, "addressee_value", section, value_words, ratio))
                else:
                    pending_addressee_row = row_idx

    return events


def _label_value_candidates(
    rows: list[list[EnsembleWord]], skip_row_idx: int | None = None
) -> list[FieldCandidate]:
    events = _collect_row_events(rows, skip_row_idx)
    number_events = [e for e in events if e.kind == "number"]
    candidates: list[FieldCandidate] = []

    for event in events:
        if event.kind not in ("name", "bank", "addressee_value"):
            continue
        text = " ".join(w.text for w in event.value_words).strip()
        if not event.value_words or _looks_numeric_or_date(text):
            continue

        if event.kind == "name":
            if event.section == "debit":
                role: field_labels.FieldRole = "debit_account_name"
            elif event.section == "credit":
                role = "credit_account_name"
            else:
                role = "counterparty_org"
        elif event.kind == "bank":
            role = "bank_name"
        else:
            role = "counterparty_org"

        account_number = None
        if event.kind == "name":
            nearest = min(
                (
                    e
                    for e in number_events
                    if e.section == event.section
                    and abs(e.row_idx - event.row_idx) <= _NUMBER_PAIR_MAX_ROW_DISTANCE
                ),
                key=lambda e: abs(e.row_idx - event.row_idx),
                default=None,
            )
            if nearest is not None:
                number_text = " ".join(w.text for w in nearest.value_words).strip()
                account_number = number_text or None

        candidates.append(
            FieldCandidate(
                start=min(w.char_start for w in event.value_words),
                end=max(w.char_end for w in event.value_words),
                entity_type="ORGANIZATION",
                field_role=role,
                match_confidence=round(event.ratio, 4),
                words=tuple(event.value_words),
                account_number=account_number,
            )
        )
    return candidates


# --- Prose "a/n" mode (free-text redemption/transfer instructions) ----


def _is_an_marker(word: EnsembleWord) -> bool:
    normalized = word.text.strip().strip(".,;").lower()
    return normalized in ("a/n", "an")


def _prose_an_candidates(ordered_words: list[EnsembleWord]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for idx, word in enumerate(ordered_words):
        if not _is_an_marker(word):
            continue
        tail = ordered_words[idx + 1 : idx + 1 + _AN_MAX_VALUE_WORDS]
        value_words: list[EnsembleWord] = []
        for w in tail:
            token = w.text.strip()
            if not token or token in (",", ".", ";", "(", ")"):
                break
            if _looks_numeric_or_date(token):
                break
            value_words.append(w)
        text = " ".join(w.text for w in value_words).strip()
        if not value_words or not text:
            continue

        role: field_labels.FieldRole = "counterparty_org"
        lookback = ordered_words[max(0, idx - _AN_LOOKBACK_WORDS) : idx]
        for lw in reversed(lookback):
            token = lw.text.strip(".,;:").lower()
            if token == "dari":
                role = "debit_account_name"
                break
            if token == "ke":
                role = "credit_account_name"
                break

        candidates.append(
            FieldCandidate(
                start=min(w.char_start for w in value_words),
                end=max(w.char_end for w in value_words),
                entity_type="ORGANIZATION",
                field_role=role,
                match_confidence=0.75,
                words=tuple(value_words),
            )
        )
    return candidates


def _is_kepada_marker(word: EnsembleWord) -> bool:
    normalized = word.text.strip().strip(".,;:").lower()
    return normalized == "kepada"


def _prose_kepada_candidates(ordered_words: list[EnsembleWord]) -> list[FieldCandidate]:
    """Catch an addressee named mid-sentence, e.g. "...instruksikan kepada
    Standard Chartered Custody untuk melakukan pemindahan dana...".

    Complements ``_label_value_candidates``'s row-anchored "kepada" match,
    which only fires when "kepada" starts a short, dedicated line
    (``_ADDRESSEE_MAX_ROW_WORDS``) — free-flowing instruction sentences bury
    the same marker mid-line, well past that row-length/start-position
    limit. Stops at a known continuation word ("untuk"/...), punctuation, or
    a numeric/date-looking token so it never captures more than the
    addressee itself. Safe under ``restrict_to_known_mappings``: an unmatched
    span is simply dropped downstream, never auto-redacted.
    """
    candidates: list[FieldCandidate] = []
    for idx, word in enumerate(ordered_words):
        if not _is_kepada_marker(word):
            continue
        tail = ordered_words[idx + 1 : idx + 1 + _KEPADA_MAX_VALUE_WORDS]
        value_words: list[EnsembleWord] = []
        for w in tail:
            token = w.text.strip()
            if not token or token in (",", ".", ";", ":", "(", ")"):
                break
            normalized = token.strip(".,;:").lower()
            if normalized in _KEPADA_STOP_WORDS:
                break
            if _looks_numeric_or_date(token):
                break
            value_words.append(w)
        text = " ".join(w.text for w in value_words).strip()
        if not value_words or not text:
            continue

        candidates.append(
            FieldCandidate(
                start=min(w.char_start for w in value_words),
                end=max(w.char_end for w in value_words),
                entity_type="ORGANIZATION",
                field_role="counterparty_org",
                match_confidence=0.75,
                words=tuple(value_words),
            )
        )
    return candidates


# --- Signature-block person-name mode ----------------------------------

# How much wider than ordinary same-line word spacing a horizontal gap
# must be before it's treated as a separate signatory column rather than
# a space between two words of the same name/title (see
# _split_row_by_x_gap).
_SIGNATURE_COLUMN_GAP_FACTOR = 4.0
_SIGNATURE_COLUMN_GAP_MIN_PX = 40


def _split_row_by_x_gap(row: list[EnsembleWord]) -> list[list[EnsembleWord]]:
    """Split an already y-clustered row into separate column groups when a
    horizontal gap between adjacent words is far wider than ordinary
    word-to-word spacing.

    ``_group_rows`` clusters purely by vertical adjacency, with no
    horizontal-continuity check at all — two independent signatories
    printed side by side on the same visual baseline (``Authorized by``
    blocks routinely lay out 2-3 names across one line) land in a single
    row alongside each other. Left unsplit, that combined row can exceed
    ``_SIGNATURE_MAX_WORDS``/``_SIGNATURE_ORG_MAX_WORDS`` (observed
    directly: two 2-word names plus a 1-word one hit 5 tokens) and drop
    every signatory on that line from detection at once — worse than
    merely mislabeling them as one name, this was a real ``repii/``
    sample turning "Wahyu Wijaya" / "Mira Octora S" into zero candidates.

    Only used by the signature-block detectors below: elsewhere (table
    columns, label:value rows) a wide same-row gap is already handled by
    each mode's own column/zone logic, which this must not change.
    """
    if len(row) < 2:
        return [row] if row else []
    ordered = sorted(row, key=lambda w: w.bbox.x)
    heights = sorted(w.bbox.h for w in ordered if w.bbox.h > 0)
    typical_h = heights[len(heights) // 2] if heights else _SIGNATURE_COLUMN_GAP_MIN_PX
    threshold = max(typical_h * _SIGNATURE_COLUMN_GAP_FACTOR, _SIGNATURE_COLUMN_GAP_MIN_PX)
    groups: list[list[EnsembleWord]] = [[ordered[0]]]
    for w in ordered[1:]:
        prev = groups[-1][-1]
        gap = w.bbox.x - (prev.bbox.x + prev.bbox.w)
        if gap > threshold:
            groups.append([w])
        else:
            groups[-1].append(w)
    return groups


def _looks_like_name_token(token: str) -> bool:
    if not token or not token[0].isupper():
        return False
    return all(ch.isalpha() or ch in ("'", "-") for ch in token)


def _row_tokens(row: list[EnsembleWord]) -> list[str]:
    return [w.text.strip(".,") for w in row]


def _is_signature_closer_row(tokens: list[str]) -> bool:
    lowered = " ".join(t.lower() for t in tokens)
    return any(stop in lowered for stop in field_labels.SIGNATURE_CLOSERS)


def _row_has_leading_label(row: list[EnsembleWord], group: list[EnsembleWord]) -> bool:
    """True when some other word earlier on the same row ends with a colon.

    A ``Label: value`` row (e.g. ``Account Name: Fee Agent Penjual``) is
    never how a signature block's own printed name/org line is laid out
    — but a name-shaped field *value*, printed right after its label on
    one row, can otherwise look identical to a signatory name/org line
    to ``_signature_candidates``/``_signature_org_candidates`` below. It
    only takes that value sitting in the bottom-of-page band — true on
    any letter with extra blank space beneath its real signature block —
    to have it mistaken for one (a real regression: ``Fee Agent
    Penjual``, an "Account Name:" value, got boxed as a signatory here).
    Restricted to *this* row rather than the whole page since a label on
    a different row/line is no signal about this one.
    """
    group_ids = {id(w) for w in group}
    group_left = min(w.bbox.x for w in group)
    return any(
        id(w) not in group_ids and w.bbox.x < group_left and w.text.rstrip().endswith(":")
        for w in row
    )


def _is_org_shaped_row(tokens: list[str]) -> bool:
    if not tokens:
        return False
    lowered_tokens = [t.lower() for t in tokens]
    lowered_row = " ".join(lowered_tokens)
    if any(stop in lowered_row for stop in field_labels.JOB_TITLE_STOPWORDS):
        return False
    if _is_signature_closer_row(tokens):
        return False
    return any(t in field_labels.ORG_PREFIX_STOPWORDS for t in lowered_tokens)


def _signature_org_candidates(
    rows: list[list[EnsembleWord]], page_height: float
) -> list[FieldCandidate]:
    """Company-name line in the signature block (e.g. below ``Hormat Kami,``).

    ``_signature_candidates`` is person-only and *deliberately* skips rows
    that contain an org prefix (``PT``, ``Life``, ``Insurance``, ...) so
    it never mistakes the letterhead/company line for a signatory. That
    skip left the company name fully visible. This counterpart emits
    those rows as ``ORGANIZATION`` / ``counterparty_org`` so a known
    mapping (``PT BNI Life Insurance`` → ``ORG_02``) can cover them.
    Restricted-mode lookup still drops anything that isn't already in
    the dictionary.
    """
    if page_height <= 0:
        return []
    threshold_y = page_height * (1 - _SIGNATURE_BOTTOM_FRACTION)
    candidates: list[FieldCandidate] = []
    for row in rows:
        if not row:
            continue
        row_top = min(w.bbox.y for w in row)
        if row_top < threshold_y:
            continue
        for group in _split_row_by_x_gap(row):
            if not (_SIGNATURE_ORG_MIN_WORDS <= len(group) <= _SIGNATURE_ORG_MAX_WORDS):
                continue
            if _row_has_leading_label(row, group):
                continue
            tokens = _row_tokens(group)
            if not _is_org_shaped_row(tokens):
                continue
            if any(_looks_numeric_or_date(t) for t in tokens):
                continue
            candidates.append(
                FieldCandidate(
                    start=min(w.char_start for w in group),
                    end=max(w.char_end for w in group),
                    entity_type="ORGANIZATION",
                    field_role="counterparty_org",
                    match_confidence=0.7,
                    words=tuple(group),
                )
            )
    return candidates


def _signature_candidates(
    rows: list[list[EnsembleWord]], page_height: float
) -> list[FieldCandidate]:
    if page_height <= 0:
        return []
    threshold_y = page_height * (1 - _SIGNATURE_BOTTOM_FRACTION)
    candidates: list[FieldCandidate] = []
    for row in rows:
        if not row:
            continue
        row_top = min(w.bbox.y for w in row)
        if row_top < threshold_y:
            continue
        for group in _split_row_by_x_gap(row):
            if not (_SIGNATURE_MIN_WORDS <= len(group) <= _SIGNATURE_MAX_WORDS):
                continue
            if _row_has_leading_label(row, group):
                continue
            tokens = _row_tokens(group)
            if not all(_looks_like_name_token(t) for t in tokens):
                continue
            if _is_signature_closer_row(tokens):
                continue
            lowered_tokens = [t.lower() for t in tokens]
            lowered_row = " ".join(lowered_tokens)
            if any(stop in lowered_row for stop in field_labels.JOB_TITLE_STOPWORDS):
                continue
            if any(t in field_labels.ORG_PREFIX_STOPWORDS for t in lowered_tokens):
                continue
            candidates.append(
                FieldCandidate(
                    start=min(w.char_start for w in group),
                    end=max(w.char_end for w in group),
                    entity_type="PERSON",
                    field_role="signatory_person",
                    match_confidence=0.7,
                    words=tuple(group),
                )
            )
    return candidates


def build_rows(ensemble_words: list[EnsembleWord]) -> list[list[EnsembleWord]]:
    """Split multi-word OCR boxes and cluster into visual rows.

    Exposed beyond ``extract_field_candidates``'s own internal use so
    another geometry-only detector that needs this pipeline's row
    structure — e.g. ``app.services.pii.signature_zones``, which anchors
    signature-ink zones on the same signature-block rows found below —
    doesn't reimplement word-splitting/row-clustering with its own tuned
    constants that could silently drift from these. Docling-cell
    stitching (``_stitch_docling_cells``) is deliberately not included
    here: it only matters for table cells, never for the bottom-of-page
    signature block this is meant for.
    """
    if not ensemble_words:
        return []
    return _group_rows(_split_multiword_tokens(ensemble_words))


def find_signature_anchor_bboxes(
    rows: list[list[EnsembleWord]], page_height: float
) -> list[BBox]:
    """Bounding box of every bottom-of-page signatory name/org row.

    Anchors for ``app.services.pii.signature_zones``'s ink-gap detection:
    the same rows ``_signature_candidates``/``_signature_org_candidates``
    already recognize as a signatory's printed name or the company-name
    line in a signature block, returned as plain geometry rather than
    ``FieldCandidate``\\s — the caller only needs "where is the printed
    anchor text" to locate the blank ink gap beside it, not a redaction
    decision (entity type, mock resolution, ...), which those two
    functions still own exclusively.
    """
    boxes: list[BBox] = []
    for cand in (*_signature_candidates(rows, page_height), *_signature_org_candidates(rows, page_height)):
        box = union_bbox(list(cand.words))
        if box is not None:
            boxes.append(box)
    return boxes


# --- Public entry point -------------------------------------------------


def _dedupe_candidates(candidates: list[FieldCandidate]) -> list[FieldCandidate]:
    """Keep the highest-confidence, earliest-starting candidate per word.

    Overlap is judged by shared underlying ``EnsembleWord`` objects (by
    identity), not by ``start``/``end`` char-range containment: a
    candidate's char range is only a min/max bound over its (possibly
    scattered, non-contiguous) words, so two candidates covering entirely
    different physical words can still have overlapping char ranges by
    coincidence — comparing ranges would wrongly drop one of them.
    """
    ordered = sorted(candidates, key=lambda c: (c.start, -c.match_confidence))
    result: list[FieldCandidate] = []
    claimed: set[int] = set()
    for cand in ordered:
        word_ids = {id(w) for w in cand.words}
        if word_ids & claimed:
            continue
        result.append(cand)
        claimed |= word_ids
    return result


def extract_field_candidates(
    merged_text: str,
    ensemble_words: list[EnsembleWord],
    blocks: list[DocBlock] | None = None,
    word_context: object | None = None,
) -> list[FieldCandidate]:
    """Detect name-shaped PII candidates anchored to document structure.

    Combines four geometry-only detection modes: table columns identified
    by header text, label:value pairs (with a debit/credit section state
    machine), prose ``a/n`` markers, and bottom-of-page signature blocks
    (person names and the company-name line under a closer such as
    ``Hormat Kami``).
    Detection relies on ``ensemble_words`` geometry for everything, so it
    keeps working when Docling is unavailable/degraded; see the module
    docstring for the one, purely-additive exception (multi-line table-cell
    stitching) when Docling ``cell`` blocks are supplied.

    Args:
        merged_text: Unused directly; ``start``/``end`` on results are
            offsets into this same string via ``ensemble_words`` char
            spans. Never sliced or logged here.
        ensemble_words: Aligned OCR words for one page, with bbox and
            char_start/char_end already set.
        blocks: Optional Docling structure blocks for this page. When
            ``cell``-type blocks are present, used only to stitch together
            multi-line-wrapped table cells before row-grouping runs (see
            ``_stitch_docling_cells``); column semantics still come
            entirely from this module's own heuristics. Safe to omit or
            pass ``None``/``[]`` — detection then falls back to pure
            geometry, identical to v1 behavior.
        word_context: Accepted for interface stability; not required by
            the v1 geometry-only heuristics.

    Returns:
        Non-overlapping ``FieldCandidate`` list, highest confidence wins
        on overlap. Empty list if ``ensemble_words`` is empty.
    """
    del merged_text, word_context
    if not ensemble_words:
        return []

    split_words = _split_multiword_tokens(ensemble_words)
    rows = _group_rows(split_words)
    cell_ids: dict[int, int] = {}
    if blocks:
        cells = [b for b in blocks if b.block_type == "cell"]
        # Docling/img2table often over-segment a multi-line logical cell into
        # one sub-cell per line.  Merge those sub-cells before using them as a
        # row-stitching signal, otherwise the stitcher sees each line as a
        # different cell and leaves fragments unmerged.
        cells = _merge_oversegmented_cells(cells)
        cell_ids = _assign_cell_ids(split_words, cells)
        rows = _stitch_docling_cells(rows, cell_ids, cells)
    ordered_words = sorted(split_words, key=lambda w: w.char_start)
    page_height = max((w.bbox.y + w.bbox.h for w in ensemble_words), default=0)

    header = _detect_table_header(rows)
    # OCR-geometry fallback for wrapped table cells Docling's own cell
    # geometry missed (see _column_band_stitch) — a no-op when Docling
    # already stitched everything, so this never changes behavior on
    # clean input.
    rows = _column_band_stitch(rows, header, cell_ids)
    header_idx = header[0] if header is not None else None

    candidates: list[FieldCandidate] = []
    candidates.extend(_table_column_candidates(rows, header, cell_ids))
    candidates.extend(_label_value_candidates(rows, header_idx))
    candidates.extend(_prose_an_candidates(ordered_words))
    candidates.extend(_prose_kepada_candidates(ordered_words))
    candidates.extend(_signature_candidates(rows, page_height))
    candidates.extend(_signature_org_candidates(rows, page_height))

    # Runs across *all* detection modes combined, not just table columns:
    # the same physical cell's wrap-line fragments can independently
    # surface from more than one mode (e.g. table-column zone detection
    # and the label:value fallback both matching the same "Bank" column
    # text), each assembling a different partial word subset that
    # ``_dedupe_candidates``'s word-identity check alone wouldn't
    # recognize as the same value.
    candidates = _merge_adjacent_same_column_candidates(candidates)
    deduped = _dedupe_candidates(candidates)
    logger.info(
        "field candidates extracted",
        extra={
            "candidate_count": len(deduped),
            "role_counts": {
                role: sum(1 for c in deduped if c.field_role == role)
                for role in (
                    "debit_account_name",
                    "credit_account_name",
                    "counterparty_org",
                    "bank_name",
                    "signatory_person",
                )
            },
        },
    )
    return deduped
