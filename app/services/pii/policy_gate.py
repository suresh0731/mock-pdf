import re

DENY_PATTERNS: list[tuple[str, str]] = [
    ("US_SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("US_SSN_RAW", r"\b\d{9}\b"),
    ("NRIC", r"\b[STFGM]\d{7}[A-Z]\b"),
    ("MYKAD", r"\b\d{6}-\d{2}-\d{4}\b"),
    ("HKID", r"\b[A-Z]{1,2}\d{6}\([0-9A]\)"),
    ("NIK", r"\b\d{16}\b"),
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("PHONE", r"\b\+?\d[\d\s\-().]{7,}\d\b"),
]

LOCALE_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "sg": [("NRIC", r"\b[STFGM]\d{7}[A-Z]\b")],
    "my": [("MYKAD", r"\b\d{6}-\d{2}-\d{4}\b")],
    "hk": [("HKID", r"\b[A-Z]{1,2}\d{6}\([0-9A]\)")],
    "id": [("NIK", r"\b\d{16}\b")],
}


class PIIPolicyViolation(Exception):
    def __init__(self, pattern: str, tier: str = "tier0"):
        self.pattern = pattern
        self.tier = tier
        super().__init__(f"PII policy violation: {pattern}")


def check_deny_list(text: str, locale: str | None = None) -> str | None:
    patterns = list(DENY_PATTERNS)
    if locale and locale in LOCALE_PATTERNS:
        patterns.extend(LOCALE_PATTERNS[locale])
    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return None


def assert_tier0_safe(tokenised_text: str, locale: str | None = None) -> None:
    match = check_deny_list(tokenised_text, locale)
    if match:
        raise PIIPolicyViolation(pattern=match, tier="tier0")
