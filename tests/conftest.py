from __future__ import annotations

import numpy as np
import pytest

from landfeedback import CoupledModel, FeedbackAnalyzer


@pytest.fixture
def linear_case():
    equilibrium = np.array([1.0, 2.0])
    scales = np.array([0.5, 2.0])
    process_1 = np.array([[-1.0, 0.0], [-0.1, 0.0]])
    process_2 = np.array([[0.0, 0.25], [0.0, -0.5]])
    total = process_1 + process_2

    def p1(x):
        return process_1 @ (x - equilibrium)

    def p2(x):
        return process_2 @ (x - equilibrium)

    def tendency(x):
        return p1(x) + p2(x)

    def diffusion(x):
        return np.array([0.2 + 0.1 * (x[0] - 1.0), 0.3 - 0.05 * (x[1] - 2.0)])

    model = CoupledModel(
        state_names=["soil_moisture", "soil_temperature"],
        tendency=tendency,
        diffusion=diffusion,
        processes={"water": p1, "energy": p2},
        units=["1", "degC"],
        time_unit="day",
    )
    result = FeedbackAnalyzer(model).analyze(
        equilibrium=equilibrium,
        scales=scales,
    )
    return {
        "model": model,
        "result": result,
        "equilibrium": equilibrium,
        "scales": scales,
        "jacobian": total,
    }

