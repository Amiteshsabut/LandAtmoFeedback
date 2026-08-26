"""Numerical derivatives for deterministic and stochastic model terms."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _steps(x: FloatArray, rel_step: float, abs_step: float | None) -> FloatArray:
    if rel_step <= 0:
        raise ValueError("rel_step must be positive")
    if abs_step is not None and abs_step <= 0:
        raise ValueError("abs_step must be positive")
    base = np.maximum(np.abs(x), 1.0) * rel_step
    if abs_step is not None:
        base = np.maximum(base, abs_step)
    return base


def jacobian(
    function: Callable[[NDArray], ArrayLike],
    x: ArrayLike,
    *,
    method: str = "central",
    rel_step: float = 1e-6,
    abs_step: float | None = None,
) -> FloatArray:
    """Calculate the Jacobian of a vector function.

    Parameters
    ----------
    function:
        Callable mapping a one-dimensional input to a scalar or array output.
    x:
        Evaluation point.
    method:
        ``"central"`` (default), ``"forward"``, or ``"complex"``.
    rel_step, abs_step:
        Relative and optional minimum absolute perturbation sizes.

    Returns
    -------
    numpy.ndarray
        Array with shape ``function(x).shape + (x.size,)``.
    """

    point = np.asarray(x, dtype=float)
    if point.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.all(np.isfinite(point)):
        raise ValueError("x must contain only finite values")
    method = method.lower()
    if method not in {"central", "forward", "complex"}:
        raise ValueError("method must be 'central', 'forward', or 'complex'")

    step = _steps(point, rel_step, abs_step)
    baseline = np.asarray(function(point))
    if baseline.ndim == 0:
        baseline = baseline.reshape(1)
    if not np.all(np.isfinite(baseline)):
        raise ValueError("function returned non-finite values at x")
    output_shape = baseline.shape
    result = np.empty(output_shape + (point.size,), dtype=float)

    for column in range(point.size):
        if method == "complex":
            perturbed = point.astype(complex)
            perturbed[column] += 1j * step[column]
            value = np.asarray(function(perturbed))
            if value.shape != output_shape:
                raise ValueError("function output shape changed during differentiation")
            result[..., column] = np.imag(value) / step[column]
            continue

        plus = point.copy()
        plus[column] += step[column]
        value_plus = np.asarray(function(plus), dtype=float)
        if value_plus.shape != output_shape:
            raise ValueError("function output shape changed during differentiation")

        if method == "forward":
            result[..., column] = (value_plus - baseline) / step[column]
        else:
            minus = point.copy()
            minus[column] -= step[column]
            value_minus = np.asarray(function(minus), dtype=float)
            if value_minus.shape != output_shape:
                raise ValueError("function output shape changed during differentiation")
            result[..., column] = (value_plus - value_minus) / (2.0 * step[column])

    if not np.all(np.isfinite(result)):
        raise ValueError("calculated Jacobian contains non-finite values")
    return result


def diffusion_jacobian(
    function: Callable[[NDArray], ArrayLike],
    x: ArrayLike,
    **kwargs,
) -> FloatArray:
    """Calculate a state derivative of a vector or matrix diffusion function."""

    return jacobian(function, x, **kwargs)

