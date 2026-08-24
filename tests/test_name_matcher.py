from app.services.pii.name_matcher import best_fuzzy_match, token_sort_ratio


def test_token_sort_ratio_identical_strings_is_one():
    assert token_sort_ratio("Standard Chartered Custody", "Standard Chartered Custody") == 1.0


def test_token_sort_ratio_word_order_independent():
    assert token_sort_ratio("Chartered Standard", "Standard Chartered") == 1.0


def test_token_sort_ratio_tolerates_ocr_typo():
    ratio = token_sort_ratio("Standard Chartered Custody", "Stabli Chartered Cnartercd")
    assert ratio > 0.6


def test_token_sort_ratio_empty_string_is_zero():
    assert token_sort_ratio("", "Standard Chartered") == 0.0
    assert token_sort_ratio("Standard Chartered", "   ") == 0.0


def test_token_sort_ratio_unrelated_strings_low():
    assert token_sort_ratio("Standard Chartered Custody", "PT BNI Life Insurance") < 0.4


def test_best_fuzzy_match_picks_highest_ratio():
    choices = ["Reksa Dana Bahana", "PT BNI Life Insurance", "Standard Chartered Custody"]
    match = best_fuzzy_match("Stanard Chartered Custdy", choices, threshold=0.7)
    assert match is not None
    idx, ratio = match
    assert choices[idx] == "Standard Chartered Custody"
    assert ratio > 0.7


def test_best_fuzzy_match_returns_none_below_threshold():
    choices = ["PT BNI Life Insurance"]
    assert best_fuzzy_match("Completely Unrelated Text", choices, threshold=0.85) is None


def test_best_fuzzy_match_empty_choices_returns_none():
    assert best_fuzzy_match("Anything", [], threshold=0.5) is None
