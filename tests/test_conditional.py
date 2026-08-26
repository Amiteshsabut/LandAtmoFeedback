from __future__ import annotations

import numpy as np
import pytest


def test_conditional_composites_for_linear_model(linear_case):
    result = linear_case["result"]
    values = np.linspace(-2.0, 2.0, 101)
    standardized = np.column_stack([values, -0.5 * values])
    states = linear_case["equilibrium"] + standardized * linear_case["scales"]
    comparison = result.conditional_composites(
        states,
        conditioning="soil_moisture",
        lower_quantile=0.1,
        upper_quantile=0.9,
    )
    assert comparison.lower_count == 11
    assert comparison.upper_count == 11
    assert comparison.conditioning_state == "soil_moisture"
    assert comparison.lower_mean_state[0] < comparison.upper_mean_state[0]
    tendency = comparison.tendency_frame()
    np.testing.assert_allclose(
        tendency["lower_linear"], tendency["lower_nonlinear"], atol=1e-10
    )
    np.testing.assert_allclose(
        tendency["upper_linear"], tendency["upper_nonlinear"], atol=1e-10
    )
    assert 0 <= comparison.asymmetry_index() <= 1
    assert len(comparison.contribution_frame()) == 4


def test_conditional_input_validation(linear_case):
    result = linear_case["result"]
    with pytest.raises(ValueError, match="quantiles"):
        result.conditional_composites(
            [[1, 2], [2, 3]], conditioning=0, lower_quantile=0.9, upper_quantile=0.1
        )
    with pytest.raises(ValueError, match="shape"):
        result.conditional_composites([[1], [2]], conditioning=0)

