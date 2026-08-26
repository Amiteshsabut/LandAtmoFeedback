from __future__ import annotations

import numpy as np
import pytest

from landfeedback import diffusion_jacobian, jacobian


def nonlinear_function(x):
    return np.array([x[0] ** 2 + x[1], x[0] * x[1]])


@pytest.mark.parametrize("method", ["central", "forward", "complex"])
def test_jacobian_methods(method):
    point = np.array([2.0, 3.0])
    expected = np.array([[4.0, 1.0], [3.0, 2.0]])
    tolerance = 2e-5 if method == "forward" else 1e-8
    np.testing.assert_allclose(
        jacobian(nonlinear_function, point, method=method),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


def test_scalar_output_and_diffusion_alias():
    derivative = jacobian(lambda x: x[0] ** 2 + x[1], [2.0, 3.0])
    assert derivative.shape == (1, 2)
    np.testing.assert_allclose(derivative, [[4.0, 1.0]], atol=1e-8)
    np.testing.assert_allclose(
        diffusion_jacobian(nonlinear_function, [2.0, 3.0]),
        jacobian(nonlinear_function, [2.0, 3.0]),
    )


def test_bad_derivative_arguments():
    with pytest.raises(ValueError, match="one-dimensional"):
        jacobian(nonlinear_function, [[1.0, 2.0]])
    with pytest.raises(ValueError, match="method"):
        jacobian(nonlinear_function, [1.0, 2.0], method="unknown")
    with pytest.raises(ValueError, match="rel_step"):
        jacobian(nonlinear_function, [1.0, 2.0], rel_step=0)

