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


# Institutional/organizational *boilerplate* — words that name a corporate
# form or a generic banking/custody role rather than the entity's own
# distinguishing identity. Deliberately narrow and hand-curated (not derived
# from corpus word-frequency): a frequency-based cutoff was tried first and
# rejected — on the real ~600-row client mapping table it also flagged
# domain words like "insurance"/"assurance" as "generic" purely because
# they're common, which then collapsed unrelated organizations that only
# share a corporate-shell shape (e.g. "PT Prudential Life Assurance" vs.
# the unrelated "PT MNC Life Assurance") into false matches. This list
# contains only words that are near-universally corporate-form filler
# across *any* institution's name, never a product/fund category word
# (e.g. "insurance", "equities", "securities" stay OUT deliberately —
# "NATIXIS - EQUITIES" and "Wholesale Banking - Securities Services" show
# both can be part of a real distinguishing name, not filler).
GENERIC_ORG_TOKENS: frozenset[str] = frozenset(
    {
        "bank",
        "custody",
        "custodian",
        "kustodian",  # Indonesian spelling of "custodian"
        "ltd",
        "limited",
        "corp",
        "corporation",
        "incorporated",
        "inc",
        "company",
        "co",
        "group",
        "holding",
        "holdings",
        "wholesale",
        "banking",
        "services",
    }
)

# A stripped string shorter than this carries too little signal to compare
# safely — e.g. stripping every generic token from a candidate that was
# *only* generic tokens ("Bank Custody Services") would otherwise leave an
# empty or near-empty string that trivially "matches" almost anything.
MIN_STRIPPED_LENGTH = 6

# Bounds for treating a candidate token as "OCR noise on a specific known
# word" rather than dropping it via exact GENERIC_ORG_TOKENS membership —
# see ``strip_generic_org_tokens``'s ``fuzzy_against`` parameter. Both
# bounds matter: length difference alone would let a short, unrelated
# token like a single fund-class letter slip through; ratio alone would
# accept a same-length-but-unrelated word. Calibrated against a real OCR
# read ("Custody" -> "Cursdy", ratio 0.769, length diff 1) while checked
# not to also accept unrelated-but-similarly-shaped word pairs (see
# ``strip_generic_org_tokens``'s docstring for why a *global* version of
# this same check was rejected).
_OCR_NOISE_MAX_LEN_DIFF = 2
_OCR_NOISE_MIN_RATIO = 0.75


def _resembles_ocr_noise(token: str, known_words: frozenset[str]) -> bool:
    """True if ``token`` is a bounded, close character-level match for one
    of ``known_words`` — e.g. a single OCR-dropped/substituted letter —
    not merely an unrelated word that happens to share some characters.
    """
    for word in known_words:
        if abs(len(token) - len(word)) > _OCR_NOISE_MAX_LEN_DIFF:
            continue
        if fuzz.ratio(token, word) / 100.0 >= _OCR_NOISE_MIN_RATIO:
            return True
    return False


def strip_generic_org_tokens(
    normalized: str, *, fuzzy_against: frozenset[str] = frozenset()
) -> str:
    """Drop ``GENERIC_ORG_TOKENS`` words, keep everything else in order.

    A curated mapping table entered by different people over time rarely
    spells an institution's name the same way twice — one row calls it
    ``"Standard Chartered Bank - Custody"``, another ``"PT Standard
    Chartered Bank Custody"``, another just ``"Bank Standard Chartered"``.
    Each variant adds/reorders/drops a *generic* corporate-form or
    banking-role word around the same distinguishing name ("Standard
    Chartered"), which dilutes ``token_sort_ratio`` enough (measured
    directly: 0.88 at best across this table's own variants) to fall
    below the stricter curated-entry threshold and leave a real,
    dictionary-known organization completely unredacted. Stripping this
    narrow, hand-curated boilerplate list before comparing lets two
    spellings of the *same* name line up on their distinguishing words
    instead of being penalized for boilerplate that differs only because
    two people wrote the address block differently.

    Only whole tokens are dropped — trailing punctuation (``"Bank,"``,
    ``"Custody."``) is trimmed first so it doesn't hide a match, but a
    token embedded in a longer word (``"Banking"`` inside a compound) is
    never partially stripped.

    Args:
        normalized: Already-normalized text (see
            ``mock_dictionary.normalize_source``) — never logged.
        fuzzy_against: Extra words to also drop a token for on a close
            (not exact) character match — e.g. an OCR misread of a
            *specific known entry's own* boilerplate word, such as
            "Custody" read as "Cursdy" (0.769 ratio). Deliberately scoped
            to the caller's one candidate entry rather than
            ``GENERIC_ORG_TOKENS`` as a whole: fuzzy-matching against the
            *global* list would also flag real distinguishing words that
            merely resemble a generic one (measured directly: "Equities"
            vs. "Securities" scores an almost identical 0.778, but
            "NATIXIS - EQUITIES" needs "Equities" to survive). Requiring
            the fuzzy target to already be a token that's actually
            present on this specific entry — checked only after the
            *other* words already line up — keeps that risk out.

    Returns:
        Space-joined remaining tokens, in original order. May be empty
        if every token was generic (or close enough to ``fuzzy_against``).
    """
    kept = []
    for token in normalized.split():
        cleaned = token.strip(".,")
        if not cleaned or cleaned in GENERIC_ORG_TOKENS:
            continue
        if fuzzy_against and _resembles_ocr_noise(cleaned, fuzzy_against):
            continue
        kept.append(cleaned)
    return " ".join(kept)
