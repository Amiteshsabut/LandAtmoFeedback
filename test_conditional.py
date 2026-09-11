import numpy as np
import pytest


def test_conditional_composites_follow_fixed_equilibrium_linearization(linear_case):
    result = linear_case["result"]
    values = np.linspace(-2.0, 2.0, 101)
    standardized = np.column_stack([values, -0.5 * values])
    states = linear_case["equilibrium"] + standardized * linear_case["scales"]
    comparison = result.conditional_composites(
        states,
        conditioning="soil_moisture",
        lower_quantile=0.05,
        upper_quantile=0.95,
    )
    assert comparison.conditioning_state == "soil_moisture"
    assert comparison.lower_mean_state[0] < comparison.upper_mean_state[0]
    np.testing.assert_allclose(
        comparison.lower_linearized_tendency,
        comparison.lower_nonlinear_tendency,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        comparison.upper_linearized_tendency,
        comparison.upper_nonlinear_tendency,
        atol=1e-10,
    )
    assert comparison.lower_standardized_disequilibrium[0] < 0
    assert comparison.upper_standardized_disequilibrium[0] > 0


def test_conditional_input_validation(linear_case):
    result = linear_case["result"]
    with pytest.raises(ValueError, match="quantiles"):
        result.conditional_composites(
            [[1, 2], [2, 3]], conditioning=0, lower_quantile=0.9, upper_quantile=0.1
        )
    with pytest.raises(ValueError, match="shape"):
        result.conditional_composites([[1], [2]], conditioning=0)
