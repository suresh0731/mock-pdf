from app.services.pii.custom_redact import find_term_spans, parse_custom_terms


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
