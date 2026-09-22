"""Consolidate all per-case result fragments after batch jobs finish."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.helpers.results import consolidate_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    args = parser.parse_args()

    print(f"Consolidating result fragments in {args.results_dir} ...")
    consolidate_results(results_dir=args.results_dir)
    print("Consolidation complete.")


if __name__ == "__main__":
    main()
