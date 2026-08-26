"""Equilibrium solving and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares, root

from .model import CoupledModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EquilibriumSolution:
    """Result returned by :func:`solve_equilibrium`."""

    state: FloatArray
    residual: FloatArray
    residual_norm: float
    success: bool
    message: str
    evaluations: int | None


def verify_equilibrium(
    model: CoupledModel,
    state: ArrayLike,
    *,
    tolerance: float = 1e-8,
) -> FloatArray:
    """Return the residual and raise if a state is not an equilibrium."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    x = model.as_state(state, name="equilibrium")
    residual = model.evaluate_tendency(x)
    norm = float(np.linalg.norm(residual, ord=np.inf))
    if norm > tolerance:
        raise ValueError(
            f"equilibrium residual {norm:.3g} exceeds tolerance {tolerance:.3g}"
        )
    return residual


def solve_equilibrium(
    model: CoupledModel,
    x0: ArrayLike,
    *,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    tolerance: float = 1e-10,
    max_evaluations: int = 2000,
) -> EquilibriumSolution:
    """Solve ``G(x)=0`` from an initial state.

    Unbounded problems use ``scipy.optimize.root``. Bounded problems use
    ``scipy.optimize.least_squares``.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    initial = model.as_state(x0, name="x0")

    if bounds is None:
        raw = root(model.evaluate_tendency, initial, method="hybr", tol=tolerance)
        state = np.asarray(raw.x, dtype=float)
        success = bool(raw.success)
        message = str(raw.message)
        evaluations = getattr(raw, "nfev", None)
    else:
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        if lower.shape not in {(), initial.shape} or upper.shape not in {(), initial.shape}:
            raise ValueError("each bound must be scalar or have one value per state")
        raw = least_squares(
            model.evaluate_tendency,
            initial,
            bounds=(lower, upper),
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
            max_nfev=max_evaluations,
        )
        state = np.asarray(raw.x, dtype=float)
        success = bool(raw.success)
        message = str(raw.message)
        evaluations = getattr(raw, "nfev", None)

    residual = model.evaluate_tendency(state)
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))
    success = success and residual_norm <= max(tolerance * 10.0, 1e-12)
    if not success:
        message = f"{message}; residual infinity norm={residual_norm:.6g}"

    return EquilibriumSolution(
        state=state,
        residual=residual,
        residual_norm=residual_norm,
        success=success,
        message=message,
        evaluations=evaluations,
    )

