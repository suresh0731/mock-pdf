"""Bilingual (EN/ID) label keyword tables for field-anchored PII detection.

All phrases are lowercase; matching is case-insensitive and tolerant of
minor OCR noise via ``name_matcher.token_sort_ratio``. No PII lives here —
these are structural anchors (labels), never document content.
"""

from typing import Literal

FieldRole = Literal[
    "debit_account_name",
    "credit_account_name",
    "counterparty_org",
    "bank_name",
    "signatory_person",
]

ACCOUNT_NAME_LABELS: tuple[str, ...] = (
    "a/c name",
    "account name",
    "nama rekening",
    "nama fund",
    "fund name",
)

BANK_LABELS: tuple[str, ...] = ("bank",)

ACCOUNT_NUMBER_LABELS: tuple[str, ...] = (
    "a/c no",
    "account no",
    "account number",
    "no rekening",
    "nomor rekening",
)

DEBIT_SECTION_LABELS: tuple[str, ...] = (
    "debit dari",
    "debit rekening",
    "debit",
    "from the following account",
    "please transfer from the following account",
)

CREDIT_SECTION_LABELS: tuple[str, ...] = (
    "kredit ke",
    "kredit rekening",
    "kredit",
    "credit",
    "to the following account",
)

ADDRESSEE_LABELS: tuple[str, ...] = ("kepada", "to", "attn", "up")

PROSE_NAME_MARKERS: tuple[str, ...] = ("a/n", "atas nama")

SIGNATURE_CLOSERS: tuple[str, ...] = (
    "hormat kami",
    "authorized by",
    "acknowledged by",
    "yours sincerely",
    "demikian",
)

# Rows containing these tokens are job titles / department lines, not the
# signatory's own name (the name line sits above the title line).
JOB_TITLE_STOPWORDS: tuple[str, ...] = (
    "head",
    "manager",
    "division",
    "department",
    "director",
    "officer",
    "operations",
    "authorized",
    "acknowledged",
    "sincerely",
)

# Rows containing these tokens are organization names, not person names.
ORG_PREFIX_STOPWORDS: tuple[str, ...] = (
    "pt",
    "pt.",
    "cv",
    "bank",
    "insurance",
    "life",
    "custody",
    "corp",
    "ltd",
    "inc",
    "asuransi",
)

DATE_MONTH_WORDS: tuple[str, ...] = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "januari",
    "februari",
    "maret",
    "mei",
    "juni",
    "juli",
    "agustus",
    "oktober",
    "desember",
)
