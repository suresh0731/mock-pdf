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
    needle = term.strip()
    if not needle:
        return []
    spans: list[tuple[int, int]] = []
    lower_text = text.lower()
    lower_needle = needle.lower()
    start = 0
    while True:
        idx = lower_text.find(lower_needle, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + max(1, len(needle))
    return spans
