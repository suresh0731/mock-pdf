"""One-off/rerunnable cleanup: merge punctuation-variant duplicate rows in
a mappings CSV (e.g. ``complete_client_mappings.csv``) down to one row
each.

Why this exists: ``normalize_source`` (see app/services/pii/mock_
dictionary.py) only casefolds and collapses whitespace — it never strips
punctuation — so two rows differing only by a trailing period ("PT" vs.
"PT.") or hyphen spacing ("TMLI - INVESTMENT" vs. "TMLI -INVESTMENT")
become two *separate* dictionary entries with two different mock values.
Which one a given document's OCR happens to produce then decides which
mock value the same real client gets, breaking the "same source text
always resolves to the same mapping" guarantee.

Grouping key strips all punctuation (``loose_key``) so both spelling
variants land in the same group; within a group with more than one row,
the surviving mock value is chosen by ``_rank`` (lower ranks first):

1. A "placeholder-style" mock (``Client-000000123``-shaped: 5+ leading
   zeros, matching the sequential batch this file's own contents show
   was appended near the end, "Client-000000004".."Client-000000133")
   loses to a mock that isn't shaped that way — the placeholder batch is
   the most likely source of an accidental re-add of something already
   mapped under a "real" generated code (e.g. "Client-30681669532").
2. Otherwise, the row appearing earliest in the file wins (stable,
   deterministic — the same policy ``insert_new_rows`` already applies
   to an exact-duplicate ``source_text``).

A group where every row is on the same side of rule 1 (all "real"-style
or all placeholder-style) is resolved purely by rule 2 and is reported
separately as NEEDS MANUAL REVIEW, since the script cannot tell which of
two equally-plausible codes is actually correct in the source system —
only a human with access to that system can confirm.

Usage (from repo root):
    python scripts/dedupe_mappings_csv.py complete_client_mappings.csv
    python scripts/dedupe_mappings_csv.py complete_client_mappings.csv --dry-run
    python scripts/dedupe_mappings_csv.py complete_client_mappings.csv --out cleaned.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_PLACEHOLDER_MOCK_RE = re.compile(r"^Client-0{5,}\d+$")


def loose_key(source_text: str) -> str:
    """Casefold and strip everything but letters/digits, collapse whitespace.

    Two rows sharing this key differ only by punctuation/spacing (a
    trailing period, hyphen spacing, ...) — the same real name in the
    source system, not two different clients.
    """
    stripped = re.sub(r"[^a-z0-9]+", " ", source_text.casefold())
    return " ".join(stripped.split())


@dataclass(frozen=True)
class _Row:
    line_no: int  # 1-based, header excluded
    source_text: str
    mock_value: str


def _is_placeholder(mock_value: str) -> bool:
    return bool(_PLACEHOLDER_MOCK_RE.match(mock_value.strip()))


def _rank(row: _Row) -> tuple[bool, int]:
    """Lower sorts first (wins). See module docstring rules 1-2."""
    return (_is_placeholder(row.mock_value), row.line_no)


def load_rows(path: Path) -> list[_Row]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [
            _Row(line_no=i, source_text=row["source_text"], mock_value=row["mock_value"])
            for i, row in enumerate(reader, start=1)
        ]


def group_duplicates(rows: list[_Row]) -> dict[str, list[_Row]]:
    groups: dict[str, list[_Row]] = {}
    for row in rows:
        groups.setdefault(loose_key(row.source_text), []).append(row)
    return {key: members for key, members in groups.items() if len(members) > 1}


def dedupe(rows: list[_Row]) -> tuple[list[_Row], list[tuple[list[_Row], _Row, bool]]]:
    """Returns (surviving rows in original order, report).

    Each report entry is ``(group_members, winner, needs_manual_review)``.
    ``needs_manual_review`` is True when rule 1 (placeholder-style) didn't
    distinguish any member — the winner was picked by rule 2 alone among
    equally-plausible candidates.
    """
    groups = group_duplicates(rows)
    losers: set[int] = set()  # line_no of rows being dropped
    report: list[tuple[list[_Row], _Row, bool]] = []

    for members in groups.values():
        ranked = sorted(members, key=_rank)
        winner = ranked[0]
        placeholder_flags = {_is_placeholder(m.mock_value) for m in members}
        needs_review = len(placeholder_flags) == 1  # rule 1 never discriminated
        report.append((members, winner, needs_review))
        for loser in ranked[1:]:
            losers.add(loser.line_no)

    surviving = [row for row in rows if row.line_no not in losers]
    return surviving, report


def _print_report(report: list[tuple[list[_Row], _Row, bool]]) -> None:
    if not report:
        print("No punctuation-variant duplicates found.")
        return

    confident = [r for r in report if not r[2]]
    needs_review = [r for r in report if r[2]]

    print(f"{len(report)} duplicate group(s) found, {len(confident)} auto-resolved confidently, "
          f"{len(needs_review)} need manual review.\n")

    if confident:
        print("=== Auto-resolved (placeholder-style mock lost) ===")
        for members, winner, _ in confident:
            print(f"  kept: {winner.source_text!r} -> {winner.mock_value!r} (line {winner.line_no})")
            for m in members:
                if m.line_no != winner.line_no:
                    print(f"    dropped: {m.source_text!r} -> {m.mock_value!r} (line {m.line_no})")
        print()

    if needs_review:
        print("=== NEEDS MANUAL REVIEW (kept first-seen; verify against the real system) ===")
        for members, winner, _ in needs_review:
            print(f"  kept (first-seen): {winner.source_text!r} -> {winner.mock_value!r} (line {winner.line_no})")
            for m in members:
                if m.line_no != winner.line_no:
                    print(f"    dropped: {m.source_text!r} -> {m.mock_value!r} (line {m.line_no})")
        print()


def _write_csv(path: Path, rows: list[_Row]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_text", "mock_value"])
        for row in rows:
            writer.writerow([row.source_text, row.mock_value])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the mappings CSV (source_text,mock_value)")
    parser.add_argument("--out", help="Write cleaned CSV here instead of overwriting the input")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the report, write nothing"
    )
    args = parser.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    rows = load_rows(path)
    surviving, report = dedupe(rows)
    _print_report(report)

    if args.dry_run:
        print(f"(dry run) {len(rows)} rows -> {len(surviving)} rows. Nothing written.")
        return

    out_path = Path(args.out) if args.out else path
    _write_csv(out_path, surviving)
    print(f"{len(rows)} rows -> {len(surviving)} rows. Wrote {out_path}")


if __name__ == "__main__":
    main()
