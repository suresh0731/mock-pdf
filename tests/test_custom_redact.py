from app.services.pii.custom_redact import find_fuzzy_term_spans, find_term_spans, parse_custom_terms


def test_parse_custom_terms_plain():
    terms = parse_custom_terms("Acme Corp\nJohn Doe")
    assert len(terms) == 2
    assert terms[0].search_value == "Acme Corp"
    assert terms[1].mock_label == "CUSTOM"


def test_parse_custom_terms_with_label():
    terms = parse_custom_terms("John Smith=MOCK_NAME")
    assert terms[0].search_value == "John Smith"
    assert terms[0].mock_label == "MOCK_NAME"


def test_find_term_spans_case_insensitive():
    text = "Contact NRIC S1234567A for billing"
    spans = find_term_spans(text, "s1234567a")
    assert len(spans) == 1
    assert text[spans[0][0] : spans[0][1]] == "S1234567A"


def test_find_term_spans_requires_token_boundary():
    """A short alias must not match inside a longer token."""
    assert find_term_spans("SCBANK Jakarta", "SCB") == []
    assert find_term_spans("Bank SCB, Jakarta", "SCB") == [(5, 8)]


def test_find_term_spans_allows_nested_phrase():
    """Word-boundary matching still finds a short phrase inside a longer one;
    longest-match claiming in the pipeline is what drops the nested hit."""
    text = "Standard Chartered Bank"
    spans = find_term_spans(text, "Standard Chartered")
    assert spans == [(0, 18)]


# --- find_fuzzy_term_spans ---------------------------------------------


def test_find_fuzzy_term_spans_finds_exact_text_too():
    text = "Hormat Kami, PT BNI Life Insurance i-asiari N"
    spans = find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE")
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "PT BNI Life Insurance"


def test_find_fuzzy_term_spans_tolerates_ocr_garble():
    """Real page-2 OCR misread: 'BNI' -> 'BNf', 'Insurance' -> 'Ins\ufffdrance'."""
    text = "Hormat Kami, PT BNf Life Ins\ufffdrance i-asiari N Dion Wicaksono"
    spans = find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE")
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "PT BNf Life Ins\ufffdrance"


def test_find_fuzzy_term_spans_no_match_for_unrelated_text():
    text = "Standard Chartered Custody, account number 12345"
    assert find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE") == []


def test_find_fuzzy_term_spans_respects_threshold():
    text = "Hormat Kami, PT BNf Life Ins\ufffdrance i-asiari N"
    # The real similarity here is ~0.90; an unreachably high threshold
    # must reject it even though a lower one accepts it.
    assert find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE", threshold=0.99) == []
    assert find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE", threshold=0.65) != []


def test_find_fuzzy_term_spans_rejects_short_alias():
    """A short alias like 'SCB' must never be fuzzy-matched — too little
    character-level signal, so an unrelated short fragment elsewhere on
    the page (e.g. inside an invoice number) can clear even a fairly
    strict threshold purely by chance. Real, observed false positive:
    'SCB' vs 'S.B' inside '0002S.BLI.SIV.INV.0126' scored 0.667."""
    text = "Invoice No: 0002S.BLI.SIV.INV.0126"
    assert find_fuzzy_term_spans(text, "SCB", threshold=0.5) == []


def test_find_fuzzy_term_spans_blank_inputs():
    assert find_fuzzy_term_spans("", "PT BNI LIFE INSURANCE") == []
    assert find_fuzzy_term_spans("some text", "") == []
    assert find_fuzzy_term_spans("   ", "PT BNI LIFE INSURANCE") == []


def test_find_fuzzy_term_spans_finds_multiple_distinct_garbled_occurrences():
    """A real page can misread the same curated name two different ways
    in two different spots (e.g. two table rows) — both must be found in
    one call, not just whichever scores highest overall."""
    text = (
        "Row 1: PT BN Life Insurance, Row 2: PT 8NILIFE INSURANCE, "
        "unrelated filler text in between and around these rows."
    )
    spans = find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE")
    matched = [text[s:e] for s, e in spans]
    assert any("PT BN Life Insurance" in m for m in matched)
    assert any("PT 8NILIFE INSURANCE" in m for m in matched)
    assert len(spans) == 2


def test_find_fuzzy_term_spans_respects_max_matches():
    text = "PT BN Life Insurance ... PT 8NILIFE INSURANCE ... PT BNi Life lnsurance"
    assert len(find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE", max_matches=1)) == 1
    assert len(find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE", max_matches=10)) >= 2


def test_find_fuzzy_term_spans_does_not_swallow_adjacent_unrelated_word():
    """Real page-2 OCR: the letterhead reads '...10 Floor PT BII UFE
    NSURANCE Phone...'. partial_ratio_alignment's best-aligned window
    started on the trailing 'r' of 'Floor', one character before the
    real match — words_for_span maps by char-range *overlap*, so that
    lone straddling character would otherwise drag the entire unrelated
    'Floor' word's bbox into the redaction (rendered as a blank white
    patch with no mock label, since it's a secondary line-wrap cluster).
    The matched span must never start or end inside a different word."""
    text = "Contennial Tower, 10 Floor PT BII UFE NSURANCE Phone (021) 5296 9928"
    spans = find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE")
    assert len(spans) == 1
    start, end = spans[0]
    assert "Floor" not in text[start:end]
    assert text[start:end].strip() == "PT BII UFE NSURANCE"


def test_find_fuzzy_term_spans_rejects_length_outlier():
    """A truncated substring (the trailing word dropped entirely) scores
    a perfect partial_ratio — it's a literal prefix — while covering
    barely half the term's own length. The length guard must reject
    that rather than treat a dropped trailing word as a confident match
    for the full term."""
    text = "pt bni life"
    assert find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE", max_length_deviation=0.3) == []
    # Disabling the guard lets the same truncated match through.
    assert (
        find_fuzzy_term_spans(text, "PT BNI LIFE INSURANCE", max_length_deviation=1.0)
        != []
    )
