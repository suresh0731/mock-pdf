"""Tests for scripts/dedupe_mappings_csv.py."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dedupe_mappings_csv import _Row, dedupe, loose_key, main  # noqa: E402


def _row(line_no: int, source_text: str, mock_value: str) -> _Row:
    return _Row(line_no=line_no, source_text=source_text, mock_value=mock_value)


def test_loose_key_ignores_trailing_period():
    assert loose_key("PT ASURANSI BRI LIFE") == loose_key("PT. Asuransi BRI Life")


def test_loose_key_ignores_hyphen_spacing():
    assert loose_key("DPLK TMLI - INVESTMENT") == loose_key("DPLK TMLI -INVESTMENT")


def test_loose_key_distinguishes_different_names():
    assert loose_key("BAHANA") != loose_key("REKSA DANA BAHANA PRIMAVERA 99")


def test_dedupe_merges_punctuation_variant_preferring_non_placeholder_mock():
    rows = [
        _row(1, "PT ASURANSI BRI LIFE", "Client-30681669532"),
        _row(2, "PT. Asuransi BRI Life", "Client-000000004"),
    ]
    surviving, report = dedupe(rows)
    assert [r.mock_value for r in surviving] == ["Client-30681669532"]
    assert len(report) == 1
    _, winner, needs_review = report[0]
    assert winner.mock_value == "Client-30681669532"
    assert needs_review is False


def test_dedupe_flags_ambiguous_group_for_manual_review():
    """Neither mock is placeholder-shaped (or both are) — the script can't
    algorithmically pick a winner, so it keeps first-seen and flags it."""
    rows = [
        _row(1, "PT Standard Chartered Bank Custody", "SCBC_PT"),
        _row(2, "PT. Standard Chartered Bank Custody", "DSCS_PT"),
    ]
    surviving, report = dedupe(rows)
    assert [r.mock_value for r in surviving] == ["SCBC_PT"]
    _, winner, needs_review = report[0]
    assert winner.mock_value == "SCBC_PT"
    assert needs_review is True


def test_dedupe_leaves_unrelated_rows_untouched():
    rows = [
        _row(1, "BAHANA", "Client-000000123"),
        _row(2, "REKSA DANA BAHANA PRIMAVERA 99", "FUND_A"),
    ]
    surviving, report = dedupe(rows)
    assert surviving == rows
    assert report == []


def test_dedupe_preserves_original_row_order():
    rows = [
        _row(1, "Alpha", "A1"),
        _row(2, "PT MANDIRI MANAJEMEN INVESTASI", "Client-000000006"),
        _row(3, "Beta", "B1"),
        _row(4, "PT. Mandiri Manajemen Investasi", "Client-000000113"),
    ]
    surviving, _ = dedupe(rows)
    assert [r.source_text for r in surviving] == [
        "Alpha",
        "PT MANDIRI MANAJEMEN INVESTASI",
        "Beta",
    ]


def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    csv_path = tmp_path / "mappings.csv"
    csv_path.write_text(
        "source_text,mock_value\n"
        "PT ASURANSI BRI LIFE,Client-30681669532\n"
        "PT. Asuransi BRI Life,Client-000000004\n",
        encoding="utf-8",
    )
    original = csv_path.read_text(encoding="utf-8")

    sys.argv = ["dedupe_mappings_csv.py", str(csv_path), "--dry-run"]
    main()

    assert csv_path.read_text(encoding="utf-8") == original
    out = capsys.readouterr().out
    assert "Nothing written" in out


def test_cli_writes_cleaned_csv_in_place(tmp_path):
    csv_path = tmp_path / "mappings.csv"
    csv_path.write_text(
        "source_text,mock_value\n"
        "PT ASURANSI BRI LIFE,Client-30681669532\n"
        "PT. Asuransi BRI Life,Client-000000004\n"
        "Unrelated Co,ORG_X\n",
        encoding="utf-8",
    )

    sys.argv = ["dedupe_mappings_csv.py", str(csv_path)]
    main()

    cleaned = csv_path.read_text(encoding="utf-8").splitlines()
    assert cleaned == [
        "source_text,mock_value",
        "PT ASURANSI BRI LIFE,Client-30681669532",
        "Unrelated Co,ORG_X",
    ]


def test_cli_out_option_leaves_input_untouched(tmp_path):
    csv_path = tmp_path / "mappings.csv"
    out_path = tmp_path / "cleaned.csv"
    csv_path.write_text(
        "source_text,mock_value\n"
        "PT ASURANSI BRI LIFE,Client-30681669532\n"
        "PT. Asuransi BRI Life,Client-000000004\n",
        encoding="utf-8",
    )
    original = csv_path.read_text(encoding="utf-8")

    sys.argv = ["dedupe_mappings_csv.py", str(csv_path), "--out", str(out_path)]
    main()

    assert csv_path.read_text(encoding="utf-8") == original
    assert out_path.exists()
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 2
