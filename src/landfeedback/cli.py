"""Command-line interface for package inspection and benchmark export."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import pandas as pd

from . import __version__
from .benchmarks import brubaker1996


def _tables() -> dict[str, Callable[[], pd.DataFrame]]:
    return {
        "states": brubaker1996.state_statistics,
        "parameters": brubaker1996.main_parameters,
        "moisture-steady": brubaker1996.deterministic_soil_moisture,
        "temperature-steady": brubaker1996.deterministic_soil_temperature,
        "reporting-matrix": brubaker1996.published_reporting_matrix,
        "moisture-conditional": brubaker1996.conditional_soil_moisture,
        "temperature-conditional": brubaker1996.conditional_soil_temperature,
        "moisture-stochastic": brubaker1996.stochastic_soil_moisture,
        "temperature-stochastic": brubaker1996.stochastic_soil_temperature,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="landfeedback")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="print a Brubaker 1996 table")
    benchmark.add_argument("table", choices=[*_tables(), "all"])
    benchmark.add_argument("--csv", action="store_true", help="print CSV instead of a text table")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""

    args = build_parser().parse_args(argv)
    if args.command == "benchmark":
        names = list(_tables()) if args.table == "all" else [args.table]
        for position, name in enumerate(names):
            if len(names) > 1:
                if position:
                    print()
                print(f"[{name}]")
            table = _tables()[name]()
            print(table.to_csv(index=True) if args.csv else table.to_string(index=False))
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

