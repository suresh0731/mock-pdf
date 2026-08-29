from app.services.pii.name_matcher import (
    best_fuzzy_match,
    is_deliberate_family_pair,
    token_sort_ratio,
)


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


# --- is_deliberate_family_pair --------------------------------------------


def test_family_pair_hyphen_with_spaces():
    assert is_deliberate_family_pair(
        "dplk axa mandiri", "dplk axa mandiri - ppip pu"
    )


def test_family_pair_hyphen_no_leading_space():
    assert is_deliberate_family_pair("dplk tmli - investment", "dplk tmli -investment") is False
    # Neither is a prefix of the other here (different token for "-
    # investment" vs "- investment") — covered by CSV dedup, not this check.
    assert is_deliberate_family_pair("dplk tmli", "dplk tmli -investment")


def test_family_pair_order_independent():
    assert is_deliberate_family_pair("dplk axa mandiri - ppip pu", "dplk axa mandiri")


def test_family_pair_plain_concatenation_is_not_a_family():
    """The accidental-OCR-truncation shape ("Maksima" / "Maksima Plus",
    no delimiter) must never be treated as a deliberate family."""
    assert is_deliberate_family_pair("maksima", "maksima plus") is False


def test_family_pair_unrelated_strings():
    assert is_deliberate_family_pair("bahana", "reksa dana bahana primavera 99") is False


def test_family_pair_identical_strings_is_false():
    assert is_deliberate_family_pair("maksima", "maksima") is False


def test_family_pair_empty_string_is_false():
    assert is_deliberate_family_pair("", "anything") is False
    assert is_deliberate_family_pair("", "") is False


def test_family_pair_bare_trailing_hyphen_is_not_a_family():
    """A hyphen with nothing meaningful after it isn't a real suffix."""
    assert is_deliberate_family_pair("dplk axa mandiri", "dplk axa mandiri -") is False
