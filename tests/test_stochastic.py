from __future__ import annotations

import numpy as np
import pytest

from landfeedback import CoupledModel, simulate_sde


def test_vector_diffusion_simulation_is_reproducible():
    model = CoupledModel(
        state_names=["a", "b"],
        tendency=lambda x: -0.1 * x,
        diffusion=lambda x: np.array([0.2, 0.4]),
        time_unit="day",
    )
    first = simulate_sde(model, [0.0, 0.0], dt=0.1, n_steps=20, seed=42)
    second = simulate_sde(model, [0.0, 0.0], dt=0.1, n_steps=20, seed=42)
    assert first.states.shape == (1, 21, 2)
    assert first.single.shape == (21, 2)
    np.testing.assert_allclose(first.states, second.states)


def test_matrix_diffusion_and_multiple_trajectories():
    model = CoupledModel(
        state_names=["a", "b"],
        tendency=lambda x: np.zeros(2),
        diffusion=lambda x: np.eye(2),
    )
    simulation = simulate_sde(
        model, [0.0, 0.0], dt=0.01, n_steps=5, n_trajectories=3, seed=1
    )
    assert simulation.states.shape == (3, 6, 2)
    with pytest.raises(ValueError, match="n_trajectories=1"):
        _ = simulation.single


def test_simulation_validation():
    model = CoupledModel(state_names=["a"], tendency=lambda x: -x)
    with pytest.raises(ValueError, match="diffusion"):
        simulate_sde(model, [0.0], dt=0.1, n_steps=10)

