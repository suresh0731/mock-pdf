import re

from rapidfuzz import fuzz

from app.models.redact import CustomRedactTerm

# How far a fuzzy span's own matched length may deviate (as a fraction of
# the search term's length) from the term itself before it's rejected.
# ``fuzz.partial_ratio_alignment`` is deliberately biased toward the
# shorter of its two inputs, so without this guard it can occasionally
# anchor on a substring markedly shorter/longer than the real term while
# still scoring above threshold (e.g. matching only half a long
# multi-word name against unrelated filler text around it).
_FUZZY_SPAN_MAX_LENGTH_DEVIATION = 0.3

# Upper bound on how many approximate occurrences of one term
# find_fuzzy_term_spans will look for on a single page. A real document
# can repeat the same curated name many times (e.g. one column value
# across every row of an 8-row table) with more than one OCR-garbled
# variant among them — bounded mainly so one term can never spin the
# iterative re-scan indefinitely on a pathological input.
_FUZZY_SPAN_MAX_MATCHES = 10

# Placeholder swapped in for an already-considered window between
# find_fuzzy_term_spans iterations. Never alphanumeric, so it can't
# itself read as part of a later, different match.
_FUZZY_SCAN_MASK_CHAR = "\x00"

# A term shorter than this is never fuzzy-matched at all — short aliases
# ("SCB") carry too little character-level signal to fuzzy-match safely:
# an unrelated 3-character fragment elsewhere on the page (e.g. inside an
# invoice number like "0002S.BLI.SIV...") can trivially clear even a
# fairly strict similarity threshold purely by chance, since so few edits
# separate almost any two short strings. Calibrated against this
# codebase's own dictionary: it excludes "SCB" (3 chars, exact-match-only
# from here on) while still covering every real multi-word name
# (shortest is "BRI Life" / "Ms. Weni" at 8 chars).
_FUZZY_MIN_TERM_LENGTH = 8


def _trim_to_word_boundary(chars: list[str], start: int, end: int) -> tuple[int, int]:
    """Shrink ``[start, end)`` inward off any partial word it starts/ends inside.

    ``partial_ratio_alignment`` picks whichever character window best
    aligns to the search term, with no notion of where a *word* begins or
    ends — unlike ``find_term_spans``, which only ever matches at token
    boundaries. When the raw window happens to start or end mid-word (one
    observed case: it starts on the trailing "r" of an unrelated "Floor"
    right before the real match), ``words_for_span`` downstream maps by
    char-range *overlap*, so even a 1-character overlap into that word
    drags its *entire* bbox into the redaction — the OCR text is fine,
    but the painted box balloons to cover unrelated text, which reads as
    "extra padding" once rendered. This only ever shrinks the window
    (never grows it), so it can't turn an otherwise-rejected match into
    an accepted one.
    """
    while start < end and 0 < start < len(chars) and chars[start - 1].isalnum() and chars[start].isalnum():
        start += 1
    while end > start and 0 < end < len(chars) and chars[end - 1].isalnum() and chars[end].isalnum():
        end -= 1
    return start, end


def parse_custom_terms(text: str) -> list[CustomRedactTerm]:
    terms: list[CustomRedactTerm] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line:
            search, label = line.split("=", 1)
            terms.append(CustomRedactTerm(search_value=search.strip(), mock_label=label.strip() or "CUSTOM"))
        elif "|" in line:
            search, label = line.split("|", 1)
            terms.append(CustomRedactTerm(search_value=search.strip(), mock_label=label.strip() or "CUSTOM"))
        else:
            terms.append(CustomRedactTerm(search_value=line))
    return terms


def find_term_spans(text: str, term: str) -> list[tuple[int, int]]:
    """Case-insensitive spans of ``term`` in ``text``, at token boundaries.

    A short dictionary alias like ``"SCB"`` must not match inside a longer
    token (``"SCBANK"``), and ``"AGRESIF"`` must not match the interior of
    ``"AGRESIFLY"``. Punctuation around the term is allowed (``"SCB,"``).
    """
    needle = term.strip()
    if not needle:
        return []
    pattern = re.escape(needle)
    if needle[0].isalnum():
        pattern = r"(?<![A-Za-z0-9])" + pattern
    if needle[-1].isalnum():
        pattern = pattern + r"(?![A-Za-z0-9])"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]


def find_fuzzy_term_spans(
    text: str,
    term: str,
    *,
    threshold: float = 0.65,
    max_length_deviation: float = _FUZZY_SPAN_MAX_LENGTH_DEVIATION,
    max_matches: int = _FUZZY_SPAN_MAX_MATCHES,
) -> list[tuple[int, int]]:
    """Approximate spans of ``term`` in ``text``, best-scoring first.

    A fallback for OCR-garbled text that ``find_term_spans``'s exact,
    case-insensitive scan can't match (a single misread character breaks
    it entirely — e.g. "BNI" read as "BNf", or "Insurance" read with a
    garbled/replacement character in place of one letter). Backed by
    ``rapidfuzz.fuzz.partial_ratio_alignment``, which finds the
    best-aligned substring of ``text`` for ``term`` and its similarity
    score in one call — the same library (and a threshold in the same
    range as) ``mock_dictionary``'s existing fuzzy *resolution* already
    uses, just applied to *locating* a span instead of scoring one that's
    already been found some other way.

    ``partial_ratio_alignment`` itself only ever returns the single best
    alignment across its whole input, so a term repeated on the page with
    more than one distinct garbled reading (a real, observed case: the
    same curated name misread two different ways in two different table
    rows) would otherwise only ever surface its highest-scoring variant.
    This masks each accepted (or rejected-but-considered) window out with
    a placeholder and re-probes, up to ``max_matches`` times, so multiple
    distinct approximate occurrences can all be found in one call — the
    same "every occurrence, not just the first" contract
    ``find_term_spans`` already has, just approximate.

    Args:
        text: Page text to search (never logged).
        term: Dictionary/custom term to search for (never logged).
        threshold: Minimum similarity ratio (``[0.0, 1.0]``) to accept.
        max_length_deviation: Maximum fraction ``term``'s own length the
            matched span's length may deviate by (see module docstring).
        max_matches: Upper bound on how many occurrences to look for.

    Returns:
        Non-overlapping ``(start, end)`` spans in ``text``'s own
        (non-normalized) offsets, best-scoring first. ``[]`` if nothing
        clears ``threshold``, ``term`` is shorter than
        ``_FUZZY_MIN_TERM_LENGTH``, or either input is blank.
    """
    needle = term.strip()
    if not needle or not text.strip() or len(needle) < _FUZZY_MIN_TERM_LENGTH:
        return []
    needle_norm = needle.casefold()
    working = list(text.casefold())
    spans: list[tuple[int, int]] = []
    for _ in range(max_matches):
        result = fuzz.partial_ratio_alignment(
            needle_norm, "".join(working), score_cutoff=threshold * 100
        )
        if result is None:
            break
        raw_start, raw_end = result.dest_start, result.dest_end
        if raw_end - raw_start <= 0:
            break
        start, end = _trim_to_word_boundary(working, raw_start, raw_end)
        matched_len = end - start
        if matched_len > 0:
            deviation = abs(matched_len - len(needle_norm)) / len(needle_norm)
            if deviation <= max_length_deviation:
                spans.append((start, end))
        # Masked out (the untrimmed window) even when rejected by the
        # length guard above, so a single persistently-bad-but-highest-
        # scoring window can't make every remaining iteration re-find the
        # exact same rejected spot.
        for i in range(raw_start, raw_end):
            working[i] = _FUZZY_SCAN_MASK_CHAR
    return spans
