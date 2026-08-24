import pytest

from app.services.pii.policy_gate import assert_tier0_safe, PIIPolicyViolation, check_deny_list
from app.services.ocr.consensus import vote
from app.models.consensus import EngineResult


def test_deny_list_blocks_ssn():
    assert check_deny_list("SSN 123-45-6789") == "US_SSN"


def test_deny_list_blocks_nric():
    assert check_deny_list("NRIC S1234567A") == "NRIC"


def test_assert_tier0_safe_raises():
    with pytest.raises(PIIPolicyViolation):
        assert_tier0_safe("Contact: john@example.com")


def test_assert_tier0_safe_passes_tokenised():
    assert_tier0_safe("Customer <PERSON_1> tax id <NRIC_1>")


def test_consensus_majority():
    results = [
        EngineResult(engine="tesseract", text="123-45-6789", confidence=0.9),
        EngineResult(engine="easyocr", text="123-45-6789", confidence=0.85),
        EngineResult(engine="paddle", text="128-45-6789", confidence=0.7),
    ]
    out = vote(results, "US_SSN", "us")
    assert out.engines_agreed >= 2
    assert out.value == "123-45-6789"
