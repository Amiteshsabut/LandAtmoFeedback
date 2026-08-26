from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from landfeedback.plotting import plot_contributions, plot_feedback_matrix


def test_plotting_smoke(linear_case):
    result = linear_case["result"]
    state = linear_case["equilibrium"] + linear_case["scales"] * np.array([1.0, -1.0])
    matrix_axis = plot_feedback_matrix(result)
    contribution_axis = plot_contributions(result, state, target="soil_moisture")
    assert matrix_axis.get_xlabel() == "Predictor anomaly"
    assert "soil_moisture" in contribution_axis.get_ylabel()

