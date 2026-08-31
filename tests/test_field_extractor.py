"""Fixtures modeled on the 4 sample templates: label:value letter, two-column
debit/credit table, prose ``a/n`` transfer instruction, and a signature
block. Geometry-only — no Docling blocks required.
"""

from __future__ import annotations

from app.models.pii_chunk import BBox
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.structure.docling_adapter import DocBlock
from app.services.pii.field_extractor import (
    FieldCandidate,
    _char_width,
    _column_band_stitch,
    _dedupe_candidates,
    _detect_table_header,
    _group_rows,
    _looks_numeric_or_date,
    _normalize_label_candidate,
    _split_multiword_tokens,
    extract_field_candidates,
)
from app.services.pii.name_matcher import token_sort_ratio

WordSpec = tuple[str, int, int, int, int]


def _make_words(specs: list[WordSpec]) -> tuple[str, list[EnsembleWord]]:
    """Build EnsembleWords with char offsets matching a single-space merge.

    ``specs`` must already be in the intended reading order (top-to-bottom,
    left-to-right) — mirrors what ``align_word_boxes`` produces.
    """
    words: list[EnsembleWord] = []
    texts: list[str] = []
    offset = 0
    for text, x, y, w, h in specs:
        words.append(
            EnsembleWord(
                text=text,
                bbox=BBox(x=x, y=y, w=w, h=h),
                ocr_confidence=0.9,
                engine_agreement=1.0,
                engines=["tesseract"],
                char_start=offset,
                char_end=offset + len(text),
            )
        )
        texts.append(text)
        offset += len(text) + 1
    return " ".join(texts), words


def _span_text(merged: str, cand: FieldCandidate) -> str:
    return merged[cand.start : cand.end]


# --- Template 1: label:value letter (payment-order style) --------------


def test_label_value_letter_debit_credit_and_bank():
    specs: list[WordSpec] = [
        # Debit section
        ("Debit", 0, 0, 60, 20),
        ("dari", 65, 0, 40, 20),
        ("Bank", 0, 40, 60, 20),
        (":", 65, 40, 10, 20),
        ("Standard", 80, 40, 70, 20),
        ("Chartered", 155, 40, 80, 20),
        ("Custody", 240, 40, 70, 20),
        ("A/C", 0, 80, 30, 20),
        ("No", 35, 80, 30, 20),
        (":", 70, 80, 10, 20),
        ("11002345", 85, 80, 100, 20),
        ("A/C", 0, 120, 30, 20),
        ("Name", 35, 120, 50, 20),
        (":", 90, 120, 10, 20),
        ("Reksa", 105, 120, 55, 20),
        ("Dana", 165, 120, 45, 20),
        ("Bahana", 215, 120, 65, 20),
        ("Primavera", 285, 120, 80, 20),
        ("99", 370, 120, 25, 20),
        # Credit section
        ("Kredit", 0, 200, 65, 20),
        ("ke", 70, 200, 25, 20),
        ("Bank", 0, 240, 60, 20),
        (":", 65, 240, 10, 20),
        ("PT", 80, 240, 30, 20),
        ("Bank", 115, 240, 55, 20),
        ("Mandiri", 175, 240, 65, 20),
        ("A/C", 0, 280, 30, 20),
        ("No", 35, 280, 30, 20),
        (":", 70, 280, 10, 20),
        ("22009988", 85, 280, 100, 20),
        ("A/C", 0, 320, 30, 20),
        ("Name", 35, 320, 50, 20),
        (":", 90, 320, 10, 20),
        ("PT", 105, 320, 30, 20),
        ("BNI", 140, 320, 40, 20),
        ("Life", 185, 320, 40, 20),
        ("Insurance", 230, 320, 80, 20),
    ]
    merged, words = _make_words(specs)
    candidates = extract_field_candidates(merged, words)

    debit_name = next(c for c in candidates if c.field_role == "debit_account_name")
    credit_name = next(c for c in candidates if c.field_role == "credit_account_name")
    banks = [c for c in candidates if c.field_role == "bank_name"]

    assert _span_text(merged, debit_name) == "Reksa Dana Bahana Primavera 99"
    assert debit_name.account_number == "11002345"
    assert debit_name.entity_type == "ORGANIZATION"

    assert _span_text(merged, credit_name) == "PT BNI Life Insurance"
    assert credit_name.account_number == "22009988"

    bank_texts = {_span_text(merged, b) for b in banks}
    assert bank_texts == {"Standard Chartered Custody", "PT Bank Mandiri"}

    # Account numbers and the numeric "99" tail are never redacted directly.
    assert all(not _looks_numeric_or_date(_span_text(merged, c)) for c in candidates)


def test_label_value_addressee_org_on_own_line():
    specs: list[WordSpec] = [
        ("Kepada", 0, 0, 70, 20),
        ("Standard", 0, 40, 70, 20),
        ("Chartered", 75, 40, 80, 20),
        ("Custody", 160, 40, 70, 20),
    ]
    merged, words = _make_words(specs)
    candidates = extract_field_candidates(merged, words)
    org = next(c for c in candidates if c.field_role == "counterparty_org")
    assert _span_text(merged, org) == "Standard Chartered Custody"


def test_label_value_numeric_only_value_is_never_a_candidate():
    specs: list[WordSpec] = [
        ("A/C", 0, 0, 30, 20),
        ("Name", 35, 0, 50, 20),
        (":", 90, 0, 10, 20),
        ("123456789", 105, 0, 100, 20),
    ]
    merged, words = _make_words(specs)
    candidates = extract_field_candidates(merged, words)
    assert candidates == []


# --- Template 2: two-column debit/credit table (BNI Life style) --------


def test_table_column_debit_credit_by_header_position():
    header: list[WordSpec] = [
        ("No", 0, 0, 30, 20),
        ("Jumlah", 50, 0, 80, 20),
        ("Nama", 235, 0, 60, 20),
        ("Rekening", 305, 0, 60, 20),
        ("Bank", 470, 0, 60, 20),
        ("No", 685, 0, 60, 20),
        ("Rekening", 755, 0, 60, 20),
        ("Nama", 985, 0, 60, 20),
        ("Rekening", 1055, 0, 60, 20),
        ("Bank", 1220, 0, 60, 20),
        ("No", 1435, 0, 60, 20),
        ("Rekening", 1505, 0, 60, 20),
    ]
    data: list[WordSpec] = [
        ("1", 0, 100, 20, 20),
        ("302,176.00", 50, 100, 100, 20),
        ("PT", 210, 100, 30, 20),
        ("BNI", 245, 100, 35, 20),
        ("LIFE", 285, 100, 50, 20),
        ("Standard", 410, 100, 70, 20),
        ("Chartered", 490, 100, 80, 20),
        ("Custody", 575, 100, 20, 20),
        ("30608778491", 700, 100, 100, 20),
        ("Blife", 960, 100, 40, 20),
        ("Link", 1005, 100, 35, 20),
        ("Campuran", 1045, 100, 70, 20),
        ("Standard", 1160, 100, 70, 20),
        ("Chartered", 1240, 100, 80, 20),
        ("Custody", 1325, 100, 20, 20),
        ("30608788780", 1450, 100, 100, 20),
    ]
    merged, words = _make_words(header + data)
    candidates = extract_field_candidates(merged, words)

    debit = next(c for c in candidates if c.field_role == "debit_account_name")
    credit = next(c for c in candidates if c.field_role == "credit_account_name")
    banks = [c for c in candidates if c.field_role == "bank_name"]

    assert _span_text(merged, debit) == "PT BNI LIFE"
    assert debit.account_number == "30608778491"

    assert _span_text(merged, credit) == "Blife Link Campuran"
    assert credit.account_number == "30608788780"

    assert len(banks) == 2
    assert all(_span_text(merged, b) == "Standard Chartered Custody" for b in banks)

    # The row-index and amount columns are never picked up as candidates.
    all_texts = {_span_text(merged, c) for c in candidates}
    assert "1" not in all_texts
    assert not any("302,176" in t for t in all_texts)


def test_table_column_drops_numeric_token_glued_into_name_zone():
    """A leftover ``0.00`` sitting inside the name column (glued OCR that
    survived splitting) must not become part of the org candidate — that
    union would stretch the painted box into the USD cell."""
    header: list[WordSpec] = [
        ("Nama", 200, 0, 60, 20),
        ("Rekening", 270, 0, 60, 20),
        ("Bank", 470, 0, 60, 20),
    ]
    data: list[WordSpec] = [
        ("0.00", 190, 100, 30, 20),
        ("PT", 225, 100, 30, 20),
        ("BNI", 260, 100, 35, 20),
        ("LIFE", 300, 100, 50, 20),
        ("Some", 460, 100, 45, 20),
        ("Bank", 510, 100, 45, 20),
    ]
    merged, words = _make_words(header + data)
    candidates = extract_field_candidates(merged, words)
    debit = next(c for c in candidates if c.field_role == "debit_account_name")
    assert _span_text(merged, debit) == "PT BNI LIFE"
    assert all(w.text != "0.00" for w in debit.words)


def test_table_column_header_survives_stray_tokens_before_bank_label():
    """A stray pipe glyph plus a duplicate/garbled "Rekening" reading right
    before "Bank" must not swallow the real "Bank" header token into a
    bogus "no rekening" match.

    Regression for a real scanned-document bug: ``token_sort_ratio`` is
    order-independent, so a two-word window like "Rekening Bank" scores
    ~0.83 against the "no rekening" label purely from the shared word
    "rekening" (present in every header label here) — high enough to have
    cleared the old 0.72 header threshold. That silently ate the bank_name
    column and, worse, could hand a downstream name column the wrong
    x-position for its account-number pairing.
    """
    header: list[WordSpec] = [
        ("No", 0, 0, 30, 20),
        ("Jumlah", 50, 0, 80, 20),
        ("Nama", 235, 0, 60, 20),
        ("Rekening", 305, 0, 60, 20),
        ("|", 390, 0, 10, 20),
        ("Rekenlng", 405, 0, 55, 20),  # garbled duplicate reading, not the real cell
        ("Bank", 470, 0, 60, 20),
        ("No", 685, 0, 60, 20),
        ("Rekening", 755, 0, 60, 20),
        ("Nama", 985, 0, 60, 20),
        ("Rekening", 1055, 0, 60, 20),
        ("|", 1140, 0, 10, 20),
        ("Rekenlng", 1155, 0, 55, 20),
        ("Bank", 1220, 0, 60, 20),
        ("No", 1435, 0, 60, 20),
        ("Rekening", 1505, 0, 60, 20),
    ]
    data: list[WordSpec] = [
        ("1", 0, 100, 20, 20),
        ("302,176.00", 50, 100, 100, 20),
        ("PT", 210, 100, 30, 20),
        ("BNI", 245, 100, 35, 20),
        ("LIFE", 285, 100, 50, 20),
        ("Standard", 410, 100, 70, 20),
        ("Chartered", 490, 100, 80, 20),
        ("Custody", 575, 100, 20, 20),
        ("30608778491", 700, 100, 100, 20),
        ("Blife", 960, 100, 40, 20),
        ("Link", 1005, 100, 35, 20),
        ("Campuran", 1045, 100, 70, 20),
        ("Standard", 1160, 100, 70, 20),
        ("Chartered", 1240, 100, 80, 20),
        ("Custody", 1325, 100, 20, 20),
        ("30608788780", 1450, 100, 100, 20),
    ]
    merged, words = _make_words(header + data)
    candidates = extract_field_candidates(merged, words)

    banks = [c for c in candidates if c.field_role == "bank_name"]
    assert len(banks) == 2
    assert all(_span_text(merged, b) == "Standard Chartered Custody" for b in banks)

    debit = next(c for c in candidates if c.field_role == "debit_account_name")
    assert debit.account_number == "30608778491"
    credit = next(c for c in candidates if c.field_role == "credit_account_name")
    assert credit.account_number == "30608788780"


# --- OCR-confusion normalization for header/column label matching ------


def test_normalize_label_candidate_folds_ocr_digit_confusions():
    assert _normalize_label_candidate("8ank") == "bank"
    assert _normalize_label_candidate("N0 Rekening") == "no rekening"
    assert _normalize_label_candidate(" A/C No : ") == "a/c no"
    assert _normalize_label_candidate("acc0unt n0") == "account no"


def test_normalize_label_candidate_leaves_numeric_heavy_tokens_untouched():
    """A genuine account-number-shaped token must never be coerced toward
    looking letter-like (SEC-002 scope guard: never touches redacted value
    text, only header/column *label* comparisons)."""
    assert _normalize_label_candidate("30608778491") == "30608778491"
    assert _normalize_label_candidate("11002345") == "11002345"


def test_normalize_label_candidate_empty_and_no_op_inputs():
    assert _normalize_label_candidate("") == ""
    assert _normalize_label_candidate("   ") == ""
    assert _normalize_label_candidate("Bank") == "bank"


def test_normalize_label_candidate_raises_fuzzy_match_ratio():
    """Whichever OCR engine's reading wins a word-cluster tie-break
    (app/services/ocr/ensemble.py's align_word_boxes) shouldn't change
    whether a header label is recognized. Folding common digit/letter
    confusions before comparison strictly improves (never worsens) the
    match ratio against the true label."""
    cases = [("8ank", "bank"), ("acc0unt n0", "account no"), ("n0 rekening", "no rekening")]
    for raw, label in cases:
        raw_ratio = token_sort_ratio(raw, label)
        normalized_ratio = token_sort_ratio(_normalize_label_candidate(raw), label)
        assert normalized_ratio >= raw_ratio
    assert token_sort_ratio(_normalize_label_candidate("8ank"), "bank") == 1.0


def test_table_column_header_recognized_despite_digit_letter_ocr_confusions():
    """End-to-end: a "Bank" header column misread as "8ank" by one engine
    is still recognized and its column still redacted."""
    header: list[WordSpec] = [
        ("No", 0, 0, 30, 20),
        ("Jumlah", 50, 0, 80, 20),
        ("Nama", 235, 0, 60, 20),
        ("Rekening", 305, 0, 60, 20),
        ("8ank", 470, 0, 60, 20),  # OCR misread of "Bank" (b <-> 8)
        ("N0", 685, 0, 60, 20),  # OCR misread of "No" (o <-> 0)
        ("Rekening", 755, 0, 60, 20),
    ]
    data: list[WordSpec] = [
        ("1", 0, 100, 20, 20),
        ("302,176.00", 50, 100, 100, 20),
        ("PT", 210, 100, 30, 20),
        ("BNI", 245, 100, 35, 20),
        ("LIFE", 285, 100, 50, 20),
        ("Standard", 410, 100, 70, 20),
        ("Chartered", 490, 100, 80, 20),
        ("Custody", 575, 100, 20, 20),
        ("30608778491", 700, 100, 100, 20),
    ]
    merged, words = _make_words(header + data)
    candidates = extract_field_candidates(merged, words)

    debit = next(c for c in candidates if c.field_role == "debit_account_name")
    bank = next(c for c in candidates if c.field_role == "bank_name")
    assert _span_text(merged, debit) == "PT BNI LIFE"
    assert debit.account_number == "30608778491"
    assert _span_text(merged, bank) == "Standard Chartered Custody"


_STITCH_HEADER: list[WordSpec] = [
    ("No", 0, 0, 30, 20),
    ("Jumlah", 50, 0, 80, 20),
    ("Nama", 235, 0, 60, 20),
    ("Rekening", 305, 0, 60, 20),
    ("Bank", 470, 0, 60, 20),
    ("No", 685, 0, 60, 20),
    ("Rekening", 755, 0, 60, 20),
    ("Nama", 985, 0, 60, 20),
    ("Rekening", 1055, 0, 60, 20),
    ("Bank", 1220, 0, 60, 20),
    ("No", 1435, 0, 60, 20),
    ("Rekening", 1505, 0, 60, 20),
]

# Credit-side bank cell wrapped across 3 stacked lines (y=80/120/150 —
# consecutive-center gaps of 20-30px each, all well past _group_rows's ~12px
# tolerance for a 20px-tall word), unlike the debit-side bank cell which
# stays on one line as a control. Every other field sits on one shared
# baseline (y=100) so it forms one ordinary row regardless of stitching.
_STITCH_DATA: list[WordSpec] = [
    ("1", 0, 100, 20, 20),
    ("302,176.00", 50, 100, 100, 20),
    ("PT", 210, 100, 30, 20),
    ("BNI", 245, 100, 35, 20),
    ("LIFE", 285, 100, 50, 20),
    ("Standard", 410, 100, 70, 20),  # debit bank, single line (control)
    ("Chartered", 490, 100, 80, 20),
    ("Custody", 575, 100, 20, 20),
    ("30608778491", 700, 100, 100, 20),
    ("Blife", 960, 100, 40, 20),
    ("Link", 1005, 100, 35, 20),
    ("Campuran", 1045, 100, 70, 20),
    ("Standard", 1160, 80, 70, 20),  # credit bank, wrapped line 1
    ("Chartered", 1240, 120, 80, 20),  # credit bank, wrapped line 2
    ("Custody", 1325, 150, 20, 20),  # credit bank, wrapped line 3
    ("30608788780", 1450, 100, 100, 20),
]

# Bbox spans all 3 wrapped-line centers (x in [1195,1335], y in [90,160])
# with margin, and deliberately excludes every other word's center — the
# nearest other word (credit name "Campuran", center x=1080) sits outside
# this cell's x range.
_STITCH_CREDIT_BANK_CELL = DocBlock(
    block_id="dl-cell-0",
    block_type="cell",
    bbox=BBox(x=1140, y=70, w=250, h=110),
    text="Standard Chartered Custody",
    table_row=2,
)


def test_table_column_docling_cell_stitching_merges_wrapped_bank_cell():
    """A bank cell wrapped across 3 stacked lines is fragmented by pure
    y-adjacency row grouping into 3 separate rows — each yields its own
    single-word candidate ("Standard", "Chartered", "Custody" individually),
    none of which fuzzy-matches the full 3-word dictionary entry. Supplying
    the Docling cell bbox that spans all 3 lines lets _stitch_docling_cells
    recombine them into one row before column-zone bucketing runs, so the
    resulting candidate carries the full, correctly-ordered text.
    """
    merged, words = _make_words(_STITCH_HEADER + _STITCH_DATA)
    candidates = extract_field_candidates(merged, words, blocks=[_STITCH_CREDIT_BANK_CELL])

    banks = [c for c in candidates if c.field_role == "bank_name"]
    assert len(banks) == 2
    assert all(_span_text(merged, b) == "Standard Chartered Custody" for b in banks)

    debit = next(c for c in candidates if c.field_role == "debit_account_name")
    assert debit.account_number == "30608778491"
    credit = next(c for c in candidates if c.field_role == "credit_account_name")
    assert credit.account_number == "30608788780"


def test_table_column_docling_cell_stitching_noop_without_blocks():
    """Same wrapped-cell fixture as above, but with no Docling blocks
    supplied (``None`` and ``[]`` alike) — Docling's own cell-based
    stitching (``_stitch_docling_cells``) is unavailable without blocks,
    but the OCR-geometry column-band fallback (``_column_band_stitch``)
    recognizes the 3 stray lines as pure single-column continuations of
    the same recognized bank column and re-merges them anyway, so the
    result matches the with-blocks case: both bank cells resolve to the
    full 3-word text, never fragmented single-word candidates. Guards the
    fallback path explicitly, since ``extract_field_candidates`` must keep
    working — and stay correct — when Docling is unavailable/degraded.
    """
    merged, words = _make_words(_STITCH_HEADER + _STITCH_DATA)

    for blocks in (None, []):
        candidates = extract_field_candidates(merged, words, blocks=blocks)
        banks = [c for c in candidates if c.field_role == "bank_name"]
        bank_texts = {_span_text(merged, b) for b in banks}

        assert len(banks) == 2
        assert bank_texts == {"Standard Chartered Custody"}


# Docling/img2table sometimes over-segment a logical multi-line cell into one
# sub-cell per text line (e.g. the bank-name cell rendered as "Standard"
# + "Chartered" + "Bank" gets reported as three stacked cells).  Without
# merging those sub-cells first, _stitch_docling_cells sees three distinct
# cells and cannot glue the fragments back together.
_STITCH_CREDIT_BANK_SUBCELLS: list[DocBlock] = [
    DocBlock(
        block_id="dl-sub-0",
        block_type="cell",
        bbox=BBox(x=1140, y=70, w=250, h=35),  # "Standard" line
        text="Standard",
        table_row=2,
    ),
    DocBlock(
        block_id="dl-sub-1",
        block_type="cell",
        bbox=BBox(x=1140, y=110, w=250, h=35),  # "Chartered" line
        text="Chartered",
        table_row=3,
    ),
    DocBlock(
        block_id="dl-sub-2",
        block_type="cell",
        bbox=BBox(x=1140, y=150, w=250, h=35),  # "Custody" line
        text="Custody",
        table_row=4,
    ),
]


def test_table_column_oversegmented_cells_are_merged_before_stitching():
    """When Docling reports a wrapped logical cell as multiple stacked sub-cells,
    _merge_oversegmented_cells must reunite them before the stitcher runs, so the
    final candidate still carries the full bank-name text instead of fragments.
    """
    merged, words = _make_words(_STITCH_HEADER + _STITCH_DATA)
    candidates = extract_field_candidates(merged, words, blocks=_STITCH_CREDIT_BANK_SUBCELLS)

    banks = [c for c in candidates if c.field_role == "bank_name"]
    assert len(banks) == 2
    assert all(_span_text(merged, b) == "Standard Chartered Custody" for b in banks)

    debit = next(c for c in candidates if c.field_role == "debit_account_name")
    assert debit.account_number == "30608778491"
    credit = next(c for c in candidates if c.field_role == "credit_account_name")
    assert credit.account_number == "30608788780"


def test_column_band_stitch_noop_when_no_header_detected():
    _, words = _make_words([("Wahyu", 0, 0, 50, 20), ("Wijaya", 55, 0, 60, 20)])
    rows = _group_rows(words)
    assert _column_band_stitch(rows, None) == rows


def test_column_band_stitch_does_not_merge_across_large_vertical_gap():
    """A single-column-zone fragment far below the table (e.g. unrelated
    footer/signature text that happens to align with one column's x-range)
    must not be swept into that column's last row — only a gap consistent
    with an ordinary wrapped-cell line spacing may merge."""
    merged, words = _make_words(_STITCH_HEADER + _STITCH_DATA)
    split_words = words
    rows = _group_rows(split_words)
    header = _detect_table_header(rows)
    assert header is not None

    far_word = EnsembleWord(
        text="Custody",
        bbox=BBox(x=1160, y=900, w=70, h=20),  # same x as the wrapped bank cell, far below
        ocr_confidence=0.9,
        engine_agreement=1.0,
        engines=["tesseract"],
        char_start=len(merged) + 1,
        char_end=len(merged) + 1 + len("Custody"),
    )
    rows_with_stray = rows + [[far_word]]
    stitched = _column_band_stitch(rows_with_stray, header)

    # The far-below fragment must remain its own row, not absorbed into
    # the wrapped credit-bank cell's row far above it.
    assert any(row == [far_word] for row in stitched)


def test_table_column_header_row_itself_yields_no_candidate():
    header: list[WordSpec] = [
        ("Nama", 235, 0, 60, 20),
        ("Rekening", 305, 0, 60, 20),
        ("Bank", 470, 0, 60, 20),
        ("Nama", 985, 0, 60, 20),
        ("Rekening", 1055, 0, 60, 20),
        ("Bank", 1220, 0, 60, 20),
    ]
    data: list[WordSpec] = [
        ("Acme", 210, 100, 50, 20),
        ("Corp", 270, 100, 45, 20),
        ("Some", 460, 100, 45, 20),
        ("Bank", 510, 100, 45, 20),
        ("Beta", 960, 100, 45, 20),
        ("LLC", 1015, 100, 40, 20),
        ("Other", 1180, 100, 55, 20),
        ("Bank", 1245, 100, 45, 20),
    ]
    merged, words = _make_words(header + data)
    candidates = extract_field_candidates(merged, words)
    # No candidate's span should fall inside the header row's char range.
    header_text_len = len(" ".join(t for t, *_ in header))
    assert all(c.start > header_text_len for c in candidates)


# --- Template 3: prose "a/n" transfer instruction (BRI Life style) -----


def test_prose_an_marker_debit_and_credit_by_dari_ke():
    tokens = [
        "Harap",
        "mentransfer",
        "IDR",
        "0.00",
        "dari",
        "nomor",
        "rekening",
        "A/C",
        "No",
        "30581673319",
        ",",
        "a/n",
        "PT",
        "Asuransi",
        "BRI",
        "Life",
        ",",
        "ke",
        "rekening",
        "A/C",
        "No",
        "30581673297",
        ",",
        "a/n",
        "PT",
        "Sinarmas",
        "Life",
        ".",
    ]
    specs: list[WordSpec] = [(tok, i * 70, 0, 60, 20) for i, tok in enumerate(tokens)]
    merged, words = _make_words(specs)
    candidates = extract_field_candidates(merged, words)

    debit = next(c for c in candidates if c.field_role == "debit_account_name")
    credit = next(c for c in candidates if c.field_role == "credit_account_name")

    assert _span_text(merged, debit) == "PT Asuransi BRI Life"
    assert _span_text(merged, credit) == "PT Sinarmas Life"
    assert debit.entity_type == "ORGANIZATION"


def test_prose_an_marker_stops_before_punctuation_and_numbers():
    tokens = ["a/n", "Jane", "Doe", ",", "12345"]
    specs: list[WordSpec] = [(tok, i * 60, 0, 50, 20) for i, tok in enumerate(tokens)]
    merged, words = _make_words(specs)
    candidates = extract_field_candidates(merged, words)
    assert len(candidates) == 1
    assert _span_text(merged, candidates[0]) == "Jane Doe"


# --- Template 4: signature block ----------------------------------------


def test_signature_block_person_name_only_bottom_titlecase():
    specs: list[WordSpec] = [
        ("Invoice", 0, 0, 60, 20),
        ("Number", 65, 0, 60, 20),
        ("Hormat", 0, 900, 60, 20),
        ("kami", 65, 900, 40, 20),
        ("Wahyu", 0, 940, 50, 20),
        ("Wijaya", 55, 940, 60, 20),
        ("Division", 0, 980, 70, 20),
        ("Head", 75, 980, 45, 20),
        ("PT", 0, 1020, 30, 20),
        ("Sinarmas", 35, 1020, 70, 20),
        ("Life", 110, 1020, 40, 20),
        ("Insurance", 155, 1020, 80, 20),
    ]
    merged, words = _make_words(specs)
    candidates = extract_field_candidates(merged, words)

    signatures = [c for c in candidates if c.field_role == "signatory_person"]
    assert len(signatures) == 1
    assert _span_text(merged, signatures[0]) == "Wahyu Wijaya"
    assert signatures[0].entity_type == "PERSON"

    orgs = [c for c in candidates if c.field_role == "counterparty_org"]
    assert len(orgs) == 1
    assert _span_text(merged, orgs[0]) == "PT Sinarmas Life Insurance"
    assert orgs[0].entity_type == "ORGANIZATION"


def test_signature_block_ignored_when_not_in_bottom_region():
    specs: list[WordSpec] = [
        ("Wahyu", 0, 0, 50, 20),
        ("Wijaya", 55, 0, 60, 20),
    ]
    merged, words = _make_words(specs)
    candidates = extract_field_candidates(merged, words)
    assert candidates == []


def test_signature_block_skips_hormat_kami_closer():
    specs: list[WordSpec] = [
        ("Hormat", 0, 900, 60, 20),
        ("Kami", 65, 900, 40, 20),
    ]
    merged, words = _make_words(specs)
    candidates = extract_field_candidates(merged, words)
    assert candidates == []


def test_signature_block_two_side_by_side_signatories_both_detected():
    """Two independent signatories printed on the same visual baseline
    (a common "Authorized by" layout) must not be merged into one
    over-long row that then fails every word-count guard and drops both
    — a real repii/ sample regression this guards against."""
    specs: list[WordSpec] = [
        ("Invoice", 0, 0, 60, 20),
        ("Wahyu", 0, 900, 50, 20),
        ("Wijaya", 55, 900, 60, 20),
        ("Mira", 400, 900, 45, 20),
        ("Octora", 450, 900, 60, 20),
        ("S", 515, 900, 15, 20),
    ]
    merged, words = _make_words(specs)
    candidates = extract_field_candidates(merged, words)

    signatures = [c for c in candidates if c.field_role == "signatory_person"]
    assert {_span_text(merged, c) for c in signatures} == {"Wahyu Wijaya", "Mira Octora S"}


def test_signature_block_close_words_within_a_name_stay_one_row():
    """Ordinary inter-word spacing within a single name/title must not be
    mistaken for a column gap between two different signatories."""
    specs: list[WordSpec] = [
        ("Invoice", 0, 0, 60, 20),
        ("Wahyu", 0, 900, 50, 20),
        ("Wijaya", 62, 900, 60, 20),
    ]
    merged, words = _make_words(specs)
    candidates = extract_field_candidates(merged, words)
    signatures = [c for c in candidates if c.field_role == "signatory_person"]
    assert len(signatures) == 1
    assert _span_text(merged, signatures[0]) == "Wahyu Wijaya"


# --- General guards / dedupe --------------------------------------------


def test_extract_field_candidates_empty_words_returns_empty():
    assert extract_field_candidates("", []) == []


def test_looks_numeric_or_date_guard():
    assert _looks_numeric_or_date("") is True
    assert _looks_numeric_or_date("12345678") is True
    assert _looks_numeric_or_date("12/08/2026") is True
    assert _looks_numeric_or_date("12 August 2026") is True
    assert _looks_numeric_or_date("Standard Chartered") is False


def test_dedupe_candidates_keeps_higher_confidence_on_overlap():
    _, words = _make_words([("Acme", 0, 0, 60, 20), ("Corp", 65, 0, 60, 20)])
    low = FieldCandidate(0, 10, "ORGANIZATION", "counterparty_org", 0.5, words=tuple(words))
    high = FieldCandidate(0, 10, "ORGANIZATION", "bank_name", 0.9, words=tuple(words))
    result = _dedupe_candidates([low, high])
    assert len(result) == 1
    assert result[0].match_confidence == 0.9
    assert result[0].field_role == "bank_name"


def test_dedupe_candidates_keeps_non_overlapping():
    _, words_a = _make_words([("Acme", 0, 0, 60, 20)])
    _, words_b = _make_words([("Globex", 0, 40, 60, 20)])
    first = FieldCandidate(0, 5, "ORGANIZATION", "bank_name", 0.8, words=tuple(words_a))
    second = FieldCandidate(10, 15, "ORGANIZATION", "counterparty_org", 0.7, words=tuple(words_b))
    result = _dedupe_candidates([first, second])
    assert len(result) == 2


# --- Char-width-table tokenizer (_split_multiword_tokens) ---------------


def test_char_width_digits_narrower_than_default_and_wide_letters_wider():
    assert _char_width("4") < _char_width("a")
    assert _char_width("M") > _char_width("a")
    assert _char_width("i") < _char_width("a")
    assert _char_width(" ") < _char_width("a")


def _one_word(text: str, x: int = 0, y: int = 0, w: int = 300, h: int = 20) -> EnsembleWord:
    return EnsembleWord(
        text=text,
        bbox=BBox(x=x, y=y, w=w, h=h),
        ocr_confidence=0.9,
        engine_agreement=1.0,
        engines=["easyocr"],
        char_start=0,
        char_end=len(text),
    )


def test_split_multiword_tokens_leaves_single_word_untouched():
    word = _one_word("Standard")
    result = _split_multiword_tokens([word])
    assert result == [word]


def test_split_multiword_tokens_preserves_text_and_order():
    word = _one_word("0.00 PT BNI LIFE INSURANCE")
    result = _split_multiword_tokens([word])
    assert [w.text for w in result] == ["0.00", "PT", "BNI", "LIFE", "INSURANCE"]
    # sub-tokens stay within the source box and in left-to-right order
    for sub in result:
        assert sub.bbox.x >= word.bbox.x
        assert sub.bbox.x + sub.bbox.w <= word.bbox.x + word.bbox.w
    for earlier, later in zip(result, result[1:]):
        assert earlier.bbox.x <= later.bbox.x


def test_split_multiword_tokens_gives_short_numeric_prefix_a_narrower_share():
    """A short numeric amount glued to a long name should not be estimated
    as occupying a naive uniform-character-count share of the box — the
    digit/letter width table should give it a narrower share than a
    character-count split would (numeric prefix is short relative to its
    character count vs. the letter-heavy remainder)."""
    text = "0.00 PT BNI LIFE INSURANCE"
    word = _one_word(text, x=0, w=300)
    result = _split_multiword_tokens([word])
    amount_token = result[0]
    assert amount_token.text == "0.00"

    uniform_frac_end = len("0.00") / len(text)
    width_frac_end = (amount_token.bbox.x - word.bbox.x + amount_token.bbox.w) / word.bbox.w
    assert width_frac_end < uniform_frac_end


def test_split_multiword_tokens_interior_boundaries_do_not_cross():
    word = _one_word("Blife Link Saham Maksima")
    result = _split_multiword_tokens([word])
    for earlier, later in zip(result, result[1:]):
        assert earlier.bbox.x + earlier.bbox.w <= later.bbox.x + 1  # tolerate rounding


def test_split_multiword_tokens_char_offsets_are_valid_subranges():
    word = _one_word("Nama Rekening")
    word.char_start, word.char_end = 5, 5 + len(word.text)
    result = _split_multiword_tokens([word])
    for sub in result:
        assert word.char_start <= sub.char_start < sub.char_end <= word.char_end
