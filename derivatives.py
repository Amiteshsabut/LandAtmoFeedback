"""Numerical derivatives for deterministic and stochastic model terms."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
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

    ``central`` reproduces the basic finite-difference linearization used by
    the package. ``complex`` is provided as a high-accuracy cross-check when
    the supplied function supports complex arithmetic.
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
            if value.ndim == 0:
                value = value.reshape(1)
            if value.shape != output_shape:
                raise ValueError("function output shape changed during differentiation")
            result[..., column] = np.imag(value) / step[column]
            continue

        plus = point.copy()
        plus[column] += step[column]
        value_plus = np.asarray(function(plus), dtype=float)
        if value_plus.ndim == 0:
            value_plus = value_plus.reshape(1)
        if value_plus.shape != output_shape:
            raise ValueError("function output shape changed during differentiation")

        if method == "forward":
            result[..., column] = (value_plus - baseline) / step[column]
        else:
            minus = point.copy()
            minus[column] -= step[column]
            value_minus = np.asarray(function(minus), dtype=float)
            if value_minus.ndim == 0:
                value_minus = value_minus.reshape(1)
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
    return jacobian(function, x, **kwargs)


def derivative_convergence(
    function: Callable[[NDArray], ArrayLike],
    x: ArrayLike,
    *,
    reference: ArrayLike,
    methods: Iterable[str] = ("central", "complex"),
    rel_steps: Iterable[float] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
) -> pd.DataFrame:
    """Evaluate derivative accuracy across methods and step sizes."""

    ref = np.asarray(reference, dtype=float)
    rows: list[dict[str, float | str]] = []
    for method in methods:
        for step in rel_steps:
            estimate = jacobian(function, x, method=method, rel_step=float(step))
            if estimate.shape != ref.shape:
                raise ValueError(
                    f"reference has shape {ref.shape}, derivative has shape {estimate.shape}"
                )
            error = estimate - ref
            rows.append(
                {
                    "method": method,
                    "rel_step": float(step),
                    "max_abs_error": float(np.max(np.abs(error))),
                    "rms_error": float(np.sqrt(np.mean(error**2))),
                }
            )
    return pd.DataFrame(rows)
