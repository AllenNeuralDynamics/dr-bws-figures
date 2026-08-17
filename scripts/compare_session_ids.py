"""Compare session-table filter presets with the checked-in session ID lists.

The NWB session table used by :mod:`dr_bws.datacube` stores filter metadata in
``keywords``.  The consolidated parquet session table stores the same metadata
as named columns, so the predicates below are the parquet equivalents of the
standard presets.

Usage::

    uv run python scripts/compare_session_ids.py
    uv run python scripts/compare_session_ids.py --preset brainwide --verbose

The command exits with status 1 when any preset differs from the expected JSON
list, which makes it suitable for a check in CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_TABLE = ROOT / "assets" / "session_table.parquet"
DEFAULT_EXPECTED = ROOT / "assets" / "session_ids.json"


def session_table_presets() -> dict[str, pl.Expr]:
    """Return the standard session filters for ``session_table.parquet``.

    ``is_annotated`` is the consolidated-table counterpart of the CCF
    annotation keyword.  The project column distinguishes the brainwide and
    Templeton collections; the remaining columns directly correspond to the
    keyword predicates in ``dr_bws.datacube``.
    """

    common = (
        pl.col("is_production")
        & pl.col("is_ephys")
        & pl.col("is_task")
        & pl.col("is_annotated")
    )
    no_issues = pl.col("issues").list.len().eq(0)
    # The consolidated table exposes the positive ``is_late_autorewards``
    # flag; this is the table equivalent of not having the old
    # ``early_autorewards`` keyword.
    no_early_autorewards = pl.col("is_late_autorewards")

    return {
        "brainwide": (
            common
            & (pl.col("project") == "DynamicRouting")
            & pl.col("is_good_behavior")
            & ~pl.col("is_context_naive")
            & no_issues
            & no_early_autorewards
        ),
        "naive": (
            common
            & (pl.col("project") == "DynamicRouting")
            & pl.col("is_context_naive")
            & no_issues
            & no_early_autorewards
        ),
        "templeton": (
            common
            & pl.col("is_templeton")
            & no_issues
            & no_early_autorewards
        ),
    }


def compare_preset(
    session_table: pl.DataFrame, expected: dict[str, list[str]], preset: str
) -> tuple[set[str], set[str]]:
    """Return ``(missing, extra)`` IDs for one preset."""

    actual = set(
        session_table.filter(session_table_presets()[preset])
        .get_column("session_id")
        .unique()
        .to_list()
    )
    wanted = set(expected[preset])
    return wanted - actual, actual - wanted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-table", type=Path, default=DEFAULT_SESSION_TABLE)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--preset", choices=tuple(session_table_presets()), action="append")
    parser.add_argument("--verbose", action="store_true", help="print differing IDs")
    args = parser.parse_args()

    session_table = pl.read_parquet(args.session_table)
    # Keep this list explicit so a renamed metadata column fails clearly.
    required = {
        "session_id", "project", "issues", "is_production", "is_ephys",
        "is_task", "is_annotated", "is_good_behavior", "is_context_naive",
        "is_templeton", "is_late_autorewards",
    }
    missing_columns = sorted(required - set(session_table.columns))
    if missing_columns:
        parser.error(f"{args.session_table} is missing columns: {', '.join(missing_columns)}")

    expected = json.loads(args.expected.read_text())
    presets = args.preset or list(session_table_presets())
    failed = False
    for preset in presets:
        if preset not in expected:
            parser.error(f"{args.expected} has no list for preset {preset!r}")
        missing, extra = compare_preset(session_table, expected, preset)
        status = "MATCH" if not missing and not extra else "DIFF"
        actual_count = len(expected[preset]) - len(missing) + len(extra)
        print(f"{preset}: {status} (actual={actual_count}, expected={len(expected[preset])})")
        if args.verbose and (missing or extra):
            print(f"  missing from parquet: {sorted(missing)}")
            print(f"  extra in parquet:     {sorted(extra)}")
        failed |= bool(missing or extra)

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
