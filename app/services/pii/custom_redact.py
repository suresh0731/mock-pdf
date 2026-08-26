import re

from app.models.redact import CustomRedactTerm


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
