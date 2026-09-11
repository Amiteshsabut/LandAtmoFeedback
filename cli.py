"""Command-line interface."""

from __future__ import annotations

import argparse

from .benchmarks import brubaker1996
from .validation import full_validation_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="landfeedback")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="run the scientific verification suite")

    bench = sub.add_parser("benchmark", help="show Brubaker & Entekhabi 1996 fixtures")
    bench.add_argument(
        "table",
        choices=["states", "parameters", "deterministic", "conditional", "stochastic"],
    )

    args = parser.parse_args(argv)

    if args.command == "validate":
        report = full_validation_report()
        print(report.summary())
        return 0 if report.passed else 1

    if args.table == "states":
        print(brubaker1996.state_statistics().to_string(index=False))
    elif args.table == "parameters":
        print(brubaker1996.main_parameters().to_string(index=False))
    elif args.table == "deterministic":
        print("Table 4")
        print(brubaker1996.deterministic_soil_moisture().to_string(index=False))
        print("\nTable 5")
        print(brubaker1996.deterministic_soil_temperature().to_string(index=False))
    elif args.table == "conditional":
        print("Table 6")
        print(brubaker1996.conditional_soil_moisture().to_string(index=False))
        print("\nTable 7")
        print(brubaker1996.conditional_soil_temperature().to_string(index=False))
    else:
        print("Table 8")
        print(brubaker1996.stochastic_soil_moisture().to_string(index=False))
        print("\nTable 9")
        print(brubaker1996.stochastic_soil_temperature().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
