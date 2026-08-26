"""Complete landfeedback quick-start example."""

from __future__ import annotations

import numpy as np

from landfeedback import CoupledModel, FeedbackAnalyzer


def main() -> None:
    state_names = (
        "soil_moisture",
        "air_humidity",
        "soil_temperature",
        "air_temperature",
    )
    equilibrium = np.array([0.61, 4.25, 20.6, 15.5])
    scales = np.array([0.043, 0.39, 2.1, 1.4])

    water_matrix = np.array(
        [
            [-0.18, 0.02, -0.025, 0.010],
            [0.30, -0.35, 0.015, -0.010],
            [0.00, 0.00, 0.000, 0.000],
            [0.00, 0.00, 0.000, 0.000],
        ]
    )
    energy_matrix = np.array(
        [
            [0.00, 0.00, 0.000, 0.000],
            [0.00, 0.00, 0.000, 0.000],
            [-1.10, 0.08, -0.45, 0.220],
            [0.30, 0.05, 0.12, -0.300],
        ]
    )

    def water_budget(x: np.ndarray) -> np.ndarray:
        return water_matrix @ (x - equilibrium)

    def energy_budget(x: np.ndarray) -> np.ndarray:
        return energy_matrix @ (x - equilibrium)

    def total_tendency(x: np.ndarray) -> np.ndarray:
        return water_budget(x) + energy_budget(x)

    def wind_sensitivity(x: np.ndarray) -> np.ndarray:
        anomaly = x - equilibrium
        return np.array([0.05 - 0.02 * anomaly[0], 0.02, -0.30 - 0.03 * anomaly[2], 0.10])

    model = CoupledModel(
        state_names=state_names,
        tendency=total_tendency,
        diffusion=wind_sensitivity,
        processes={"water_budget": water_budget, "energy_budget": energy_budget},
        units=("1", "g kg-1", "degC", "degC"),
        time_unit="day",
    )

    result = FeedbackAnalyzer(model).analyze(
        equilibrium=equilibrium,
        scales=scales,
    )
    warm_dry = np.array([0.55, 4.45, 23.0, 17.0])

    print("Scaled feedback matrix")
    print(result.matrix_frame().round(3))
    print("\nWarm-dry pathway contributions")
    print(result.contribution_frame(warm_dry).round(3).to_string(index=False))
    print("\nLocal relaxation modes")
    print(result.relaxation_modes().round(3).to_string(index=False))
    print("\nMaximum absolute process-closure error")
    print(float(np.max(np.abs(result.process_closure()))))


if __name__ == "__main__":
    main()

