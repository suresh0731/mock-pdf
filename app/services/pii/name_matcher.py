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

import re

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


_FAMILY_SUFFIX_RE = re.compile(r"\s*-\s*\S")


def is_deliberate_family_pair(a: str, b: str) -> bool:
    """True when one string is the other plus a hyphen-delimited suffix.

    Some mapping tables use a "PARENT - CHILD" naming convention for
    genuinely distinct sub-accounts under one umbrella name (e.g.
    ``"dplk axa mandiri"`` / ``"dplk axa mandiri - ppip pu"``) — a
    *deliberate* family relationship, not an accidental OCR truncation
    of the same entity. Only a delimiter-bounded continuation counts:
    plain concatenation (e.g. ``"maksima"`` / ``"maksima plus"``) is
    exactly the accidental-truncation shape this must not match, since
    text shape alone can't otherwise distinguish "a truncated read of a
    longer known name" from "a deliberately shorter but equally valid
    name" — that's what ``is_token_subset_collision`` still exists to
    flag as ambiguous. This narrower, delimiter-bounded check is a
    carve-out for the specific case where the mapping table's own
    naming convention already answers the question.

    Args:
        a: First string, already normalized (see ``normalize_source``)
            — never logged.
        b: Second string, already normalized — never logged.

    Returns:
        ``True`` if the shorter is a hyphen-delimited prefix of the
        longer.
    """
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if not shorter or not longer.startswith(shorter):
        return False
    return bool(_FAMILY_SUFFIX_RE.match(longer[len(shorter) :]))


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
