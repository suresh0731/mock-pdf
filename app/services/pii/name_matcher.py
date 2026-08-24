"""Fuzzy text similarity for collapsing OCR variants of the same name.

Backed by ``rapidfuzz`` (C++ core) instead of stdlib ``difflib`` — faster
(relevant once the lookahead/lookbehind window-extension probe in
``field_extractor.py`` runs several comparisons per ambiguous candidate) and
ships ``token_set_ratio``, a well-tested primitive for the "one string's
tokens are a subset/prefix of the other's" case (e.g. ``"blife link saham
maksima"`` vs. ``"blife link saham maksima plus"``) that ``token_sort_ratio``
alone can't distinguish from ordinary character drift. Used both for label
matching (field_extractor) and for collapsing near-duplicate auto-assigned
dictionary rows (mock_dictionary). Never logs the compared text (SEC-001).
"""

from __future__ import annotations

from rapidfuzz import fuzz


def token_sort_ratio(a: str, b: str) -> float:
    """Order-independent similarity of two strings, in ``[0.0, 1.0]``.

    Tokenizes both strings, sorts tokens alphabetically, rejoins, and
    compares. Tolerant of OCR word-order noise and minor character drift.

    Args:
        a: First string (never logged).
        b: Second string (never logged).

    Returns:
        Ratio in ``[0.0, 1.0]``. ``0.0`` if either tokenizes to nothing.
    """
    if not a.strip() or not b.strip():
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def token_set_ratio(a: str, b: str) -> float:
    """Subset/containment-tolerant similarity of two strings, in ``[0.0, 1.0]``.

    Unlike ``token_sort_ratio``, this scores ``100`` whenever one string's
    token set is a subset of the other's (e.g. a truncated OCR read that's
    missing a trailing word like ``"Plus"``), regardless of length
    difference — the built-in signal for flagging a prefix/containment
    collision between two otherwise-distinct dictionary entries.

    Args:
        a: First string (never logged).
        b: Second string (never logged).

    Returns:
        Ratio in ``[0.0, 1.0]``. ``0.0`` if either tokenizes to nothing.
    """
    if not a.strip() or not b.strip():
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def is_token_subset_collision(a: str, b: str, *, min_ratio: float = 1.0) -> bool:
    """True when ``a``'s tokens are a (near-)complete subset of ``b``'s or vice versa.

    A thin, explicit wrapper around ``token_set_ratio`` for callers that
    just need a yes/no ambiguity flag (see ``mock_dictionary``'s
    prefix-collision detection) rather than the raw score. ``min_ratio``
    defaults to requiring an exact subset relationship (ratio == 1.0);
    lower it to also catch a subset relationship with a little OCR noise
    on the shared tokens.
    """
    return token_set_ratio(a, b) >= min_ratio


def best_fuzzy_match(
    candidate: str, choices: list[str], threshold: float = 0.85
) -> tuple[int, float] | None:
    """Index and ratio of the best choice at/above ``threshold``, or None.

    Args:
        candidate: Text to match (never logged).
        choices: Candidate strings to compare against (never logged).
        threshold: Minimum ratio to count as a match. Default ``0.85``.

    Returns:
        ``(index, ratio)`` of the best match, or ``None`` if none clears
        the threshold or ``choices`` is empty.
    """
    best_idx = -1
    best_ratio = 0.0
    for idx, choice in enumerate(choices):
        ratio = token_sort_ratio(candidate, choice)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx
    if best_idx == -1 or best_ratio < threshold:
        return None
    return best_idx, round(best_ratio, 4)
