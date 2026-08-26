"""Stochastic simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import CoupledModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationResult:
    """Euler-Maruyama trajectories."""

    time: FloatArray
    states: FloatArray
    state_names: tuple[str, ...]
    time_unit: str

    @property
    def n_trajectories(self) -> int:
        return self.states.shape[0]

    @property
    def single(self) -> FloatArray:
        """Return the sole trajectory or raise when several were simulated."""

        if self.n_trajectories != 1:
            raise ValueError("single is available only when n_trajectories=1")
        return self.states[0]


def simulate_sde(
    model: CoupledModel,
    x0: ArrayLike,
    *,
    dt: float,
    n_steps: int,
    n_trajectories: int = 1,
    seed: int | np.random.Generator | None = None,
) -> SimulationResult:
    """Simulate a coupled SDE using the Euler-Maruyama method.

    The model diffusion may be a vector, representing one shared Wiener
    process, or a matrix with one column per independent forcing.
    """

    if model.diffusion is None:
        raise ValueError("simulate_sde requires a model diffusion function")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive")

    initial = model.as_state(x0, name="x0")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    states = np.empty((n_trajectories, n_steps + 1, model.dimension), dtype=float)
    states[:, 0, :] = initial
    sqrt_dt = float(np.sqrt(dt))

    for trajectory in range(n_trajectories):
        for step in range(n_steps):
            current = states[trajectory, step]
            drift = model.evaluate_tendency(current)
            diffusion = model.evaluate_diffusion(current)
            if diffusion.ndim == 1:
                noise = diffusion * rng.normal() * sqrt_dt
            else:
                increments = rng.normal(size=diffusion.shape[1]) * sqrt_dt
                noise = diffusion @ increments
            states[trajectory, step + 1] = current + drift * dt + noise

    time = np.arange(n_steps + 1, dtype=float) * dt
    return SimulationResult(
        time=time,
        states=states,
        state_names=model.state_names,
        time_unit=model.time_unit,
    )

