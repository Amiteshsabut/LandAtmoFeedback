"""Reproduce the *analysis structure* and published tables used for validation."""

from pathlib import Path

from landfeedback import full_validation_report
from landfeedback.benchmarks import brubaker1996


def main():
    out = Path("validation_output")
    out.mkdir(exist_ok=True)

    print(brubaker1996.citation())
    print("\nMETHOD")
    for i, step in enumerate(brubaker1996.methodology_summary(), 1):
        print(f"{i}. {step}")

    tables = {
        "table3_state_statistics": brubaker1996.state_statistics(),
        "table4_soil_moisture_drift": brubaker1996.deterministic_soil_moisture(),
        "table5_soil_temperature_drift": brubaker1996.deterministic_soil_temperature(),
        "table6_dry_moist_soil_moisture": brubaker1996.conditional_soil_moisture(),
        "table7_dry_moist_soil_temperature": brubaker1996.conditional_soil_temperature(),
        "table8_soil_moisture_stochastic": brubaker1996.stochastic_soil_moisture(),
        "table9_soil_temperature_stochastic": brubaker1996.stochastic_soil_temperature(),
    }

    for name, table in tables.items():
        print(f"\n{name.upper()}")
        print(table.to_string(index=False))
        table.to_csv(out / f"{name}.csv", index=False)

    print("\nPUBLISHED REPORTING MATRIX")
    matrix = brubaker1996.published_reporting_matrix()
    print(matrix)
    matrix.to_csv(out / "published_reporting_matrix.csv")

    report = full_validation_report()
    print("\n" + report.summary())
    print("\nPaper arithmetic")
    print(report.paper_arithmetic.to_string(index=False))
    print("\nDerivative convergence")
    print(report.derivative_convergence.to_string(index=False))

    report.checks.to_csv(out / "validation_checks.csv", index=False)
    report.paper_arithmetic.to_csv(out / "paper_arithmetic.csv", index=False)
    report.derivative_convergence.to_csv(out / "derivative_convergence.csv", index=False)
    (out / "validation_summary.txt").write_text(report.summary(), encoding="utf-8")

    report.assert_valid()


if __name__ == "__main__":
    main()
