from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from landfeedback import FeedbackAnalyzer


def test_scaled_matrix_and_frames(linear_case):
    result = linear_case["result"]
    expected = linear_case["jacobian"] * linear_case["scales"][None, :]
    np.testing.assert_allclose(result.jacobian, linear_case["jacobian"], atol=1e-9)
    np.testing.assert_allclose(result.scaled_matrix, expected, atol=1e-9)
    assert isinstance(result.matrix_frame(), pd.DataFrame)
    assert result.matrix_frame().index.tolist() == ["soil_moisture", "soil_temperature"]
    assert result.locally_stable


def test_contributions_and_classification(linear_case):
    result = linear_case["result"]
    state = linear_case["equilibrium"] + linear_case["scales"] * np.array([1.0, -1.0])
    expected = np.array([[-0.5, -0.5], [-0.05, 1.0]])
    np.testing.assert_allclose(result.contributions(state), expected)
    assert result.classifications(state).tolist() == [
        ["restoring", "restoring"],
        ["reinforcing", "restoring"],
    ]
    frame = result.contribution_frame(state)
    assert len(frame) == 4
    assert set(frame["feedback_class"]) == {"restoring", "reinforcing"}


def test_linear_tendency_and_error(linear_case):
    result = linear_case["result"]
    state = np.array([0.8, 2.4])
    np.testing.assert_allclose(
        result.linearized_tendency(state),
        result.nonlinear_tendency(state),
        atol=1e-10,
    )
    np.testing.assert_allclose(result.linearization_error(state), 0.0, atol=1e-10)


def test_process_decomposition_and_diffusion(linear_case):
    result = linear_case["result"]
    assert set(result.process_scaled_matrices) == {"water", "energy"}
    np.testing.assert_allclose(result.process_closure(), 0.0, atol=1e-9)
    assert len(result.process_frame(target="soil_moisture")) == 4
    np.testing.assert_allclose(
        result.diffusion_jacobian,
        [[0.1, 0.0], [0.0, -0.05]],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.diffusion_scaled_sensitivity,
        [[0.05, 0.0], [0.0, -0.1]],
        atol=1e-9,
    )


def test_relaxation_and_dominant_pathways(linear_case):
    result = linear_case["result"]
    modes = result.relaxation_modes()
    assert modes["stable"].all()
    assert np.isfinite(modes["relaxation_time"]).all()
    dominant = result.dominant_pathways()
    assert dominant["target"].tolist() == ["soil_moisture", "soil_temperature"]


def test_analyzer_solves_equilibrium(linear_case):
    result = FeedbackAnalyzer(linear_case["model"]).at_equilibrium(
        x0=[0.0, 0.0],
        scales=linear_case["scales"],
    )
    np.testing.assert_allclose(result.equilibrium, linear_case["equilibrium"], atol=1e-9)


def test_bad_scales_and_equilibrium(linear_case):
    analyzer = FeedbackAnalyzer(linear_case["model"])
    with pytest.raises(ValueError, match="strictly positive"):
        analyzer.analyze(equilibrium=linear_case["equilibrium"], scales=[0.0, 1.0])
    with pytest.raises(ValueError, match="residual"):
        analyzer.analyze(equilibrium=[0.0, 0.0], scales=[1.0, 1.0])

