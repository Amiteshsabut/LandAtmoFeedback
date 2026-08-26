from __future__ import annotations

import numpy as np
import pytest

from landfeedback import CoupledModel, solve_equilibrium, verify_equilibrium


def test_unbounded_equilibrium_solve():
    target = np.array([1.5, -2.0])
    model = CoupledModel(state_names=["a", "b"], tendency=lambda x: target - x)
    solution = solve_equilibrium(model, [0.0, 0.0])
    assert solution.success
    np.testing.assert_allclose(solution.state, target, atol=1e-10)
    assert solution.residual_norm < 1e-10


def test_bounded_equilibrium_solve():
    target = np.array([0.8])
    model = CoupledModel(state_names=["a"], tendency=lambda x: target - x)
    solution = solve_equilibrium(model, [0.5], bounds=([0.0], [1.0]))
    assert solution.success
    np.testing.assert_allclose(solution.state, target, atol=1e-8)


def test_verify_equilibrium_rejects_residual():
    model = CoupledModel(state_names=["a"], tendency=lambda x: 1.0 - x)
    np.testing.assert_allclose(verify_equilibrium(model, [1.0]), [0.0])
    with pytest.raises(ValueError, match="residual"):
        verify_equilibrium(model, [0.0])

