"""Core coupled-feedback calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .derivatives import diffusion_jacobian, jacobian
from .equilibrium import solve_equilibrium, verify_equilibrium
from .model import CoupledModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FeedbackResult:
    """Feedback diagnostics evaluated around one equilibrium.

    Notes
    -----
    ``scaled_matrix`` follows Brubaker & Entekhabi (1996), Eq. (19)-(20):
    ``A_ij = (dG_i/dx_j)|x* * sigma_j``.  It therefore has units of the
    target tendency and measures the response to a one-standard-deviation
    predictor disequilibrium.  It is intentionally *not* a dimensionless
    standardized dynamical matrix.
    """

    model: CoupledModel
    equilibrium: FloatArray
    scales: FloatArray
    jacobian: FloatArray
    scaled_matrix: FloatArray
    equilibrium_residual: FloatArray
    process_jacobians: dict[str, FloatArray]
    process_scaled_matrices: dict[str, FloatArray]
    diffusion_at_equilibrium: FloatArray | None = None
    diffusion_jacobian: FloatArray | None = None
    diffusion_scaled_sensitivity: FloatArray | None = None
    derivative_method: str = "central"

    @property
    def sigma_scaled_matrix(self) -> FloatArray:
        """Alias emphasizing the Brubaker-style one-sigma scaling."""
        return self.scaled_matrix

    @property
    def standardized_dynamics_matrix(self) -> FloatArray:
        """Return ``D^-1 J D`` for standardized state dynamics.

        This is provided as an additional modern diagnostic.  It is not the
        coefficient convention used in the 1996 paper.
        """
        return (
            self.jacobian
            * self.scales[np.newaxis, :]
            / self.scales[:, np.newaxis]
        )

    def standardized_anomaly(self, state: ArrayLike) -> FloatArray:
        x = self.model.as_state(state)
        return (x - self.equilibrium) / self.scales

    def linearized_tendency(self, state: ArrayLike) -> FloatArray:
        x = self.model.as_state(state)
        return self.jacobian @ (x - self.equilibrium)

    def nonlinear_tendency(self, state: ArrayLike) -> FloatArray:
        return self.model.evaluate_tendency(state)

    def linearization_error(self, state: ArrayLike) -> FloatArray:
        return self.nonlinear_tendency(state) - self.linearized_tendency(state)

    def contributions(self, state: ArrayLike) -> FloatArray:
        """Return Eq. (20)-style pathway contributions ``A_ij * delta_j``."""
        anomaly = self.standardized_anomaly(state)
        return self.scaled_matrix * anomaly[np.newaxis, :]

    def classifications(self, state: ArrayLike, *, tolerance: float = 1e-12) -> NDArray:
        """Classify contributions as restoring/reinforcing relative to target anomaly."""
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        target_anomaly = self.standardized_anomaly(state)[:, np.newaxis]
        signed_effect = self.contributions(state) * target_anomaly
        labels = np.full(signed_effect.shape, "neutral", dtype=object)
        labels[signed_effect < -tolerance] = "restoring"
        labels[signed_effect > tolerance] = "reinforcing"
        return labels

    def matrix_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.scaled_matrix,
            index=pd.Index(self.model.state_names, name="target"),
            columns=pd.Index(self.model.state_names, name="predictor"),
        )

    def standardized_dynamics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.standardized_dynamics_matrix,
            index=pd.Index(self.model.state_names, name="target"),
            columns=pd.Index(self.model.state_names, name="predictor"),
        )

    def jacobian_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.jacobian,
            index=pd.Index(self.model.state_names, name="target"),
            columns=pd.Index(self.model.state_names, name="predictor"),
        )

    def contribution_frame(self, state: ArrayLike) -> pd.DataFrame:
        anomaly = self.standardized_anomaly(state)
        values = self.contributions(state)
        classes = self.classifications(state)
        rows = []
        for i, target in enumerate(self.model.state_names):
            for j, predictor in enumerate(self.model.state_names):
                rows.append(
                    {
                        "target": target,
                        "predictor": predictor,
                        "target_anomaly": anomaly[i],
                        "predictor_anomaly": anomaly[j],
                        "scaled_coefficient": self.scaled_matrix[i, j],
                        "contribution": values[i, j],
                        "feedback_class": classes[i, j],
                    }
                )
        return pd.DataFrame(rows)

    def process_frame(self, target: str | int | None = None) -> pd.DataFrame:
        if not self.process_scaled_matrices:
            return pd.DataFrame(
                columns=["process", "target", "predictor", "scaled_coefficient"]
            )
        targets = (
            range(self.model.dimension)
            if target is None
            else [self.model.state_index(target)]
        )
        rows = []
        for process, matrix in self.process_scaled_matrices.items():
            for i in targets:
                for j, predictor in enumerate(self.model.state_names):
                    rows.append(
                        {
                            "process": process,
                            "target": self.model.state_names[i],
                            "predictor": predictor,
                            "scaled_coefficient": matrix[i, j],
                        }
                    )
        return pd.DataFrame(rows)

    def process_closure(self) -> FloatArray:
        if not self.process_jacobians:
            return self.jacobian.copy()
        return self.jacobian - np.sum(list(self.process_jacobians.values()), axis=0)

    def relaxation_modes(self) -> pd.DataFrame:
        eigenvalues = np.linalg.eigvals(self.jacobian)
        rows = []
        for index, value in enumerate(eigenvalues, start=1):
            real = float(np.real(value))
            imag = float(np.imag(value))
            rows.append(
                {
                    "mode": index,
                    "eigenvalue_real": real,
                    "eigenvalue_imag": imag,
                    "stable": real < 0,
                    "relaxation_time": (-1.0 / real) if real < 0 else np.inf,
                    "time_unit": self.model.time_unit,
                }
            )
        return pd.DataFrame(rows)

    @property
    def locally_stable(self) -> bool:
        return bool(np.all(np.real(np.linalg.eigvals(self.jacobian)) < 0))

    def dominant_pathways(self, *, include_self: bool = False) -> pd.DataFrame:
        rows = []
        for i, target in enumerate(self.model.state_names):
            values = np.abs(self.scaled_matrix[i]).copy()
            if not include_self:
                values[i] = -np.inf
            j = int(np.argmax(values))
            rows.append(
                {
                    "target": target,
                    "predictor": self.model.state_names[j],
                    "scaled_coefficient": self.scaled_matrix[i, j],
                    "absolute_strength": abs(self.scaled_matrix[i, j]),
                }
            )
        return pd.DataFrame(rows)

    def linearized_diffusion(self, state: ArrayLike) -> FloatArray:
        """Eq. (32)-style linearization of stochastic susceptibility ``g(x)``."""
        if self.diffusion_at_equilibrium is None or self.diffusion_scaled_sensitivity is None:
            raise ValueError("model does not define diffusion")
        delta = self.standardized_anomaly(state)
        correction = np.tensordot(
            self.diffusion_scaled_sensitivity,
            delta,
            axes=([-1], [0]),
        )
        return self.diffusion_at_equilibrium + correction

    def diffusion_linearization_error(self, state: ArrayLike) -> FloatArray:
        if self.model.diffusion is None:
            raise ValueError("model does not define diffusion")
        return self.model.evaluate_diffusion(state) - self.linearized_diffusion(state)

    def conditional_composites(self, states: ArrayLike, **kwargs):
        from .conditional import conditional_composites
        return conditional_composites(self, states, **kwargs)


class FeedbackAnalyzer:
    """Construct local feedback diagnostics for a :class:`CoupledModel`."""

    def __init__(self, model: CoupledModel):
        self.model = model

    def analyze(
        self,
        *,
        equilibrium: ArrayLike,
        scales: ArrayLike,
        derivative_method: str = "central",
        rel_step: float = 1e-6,
        abs_step: float | None = None,
        jacobian_matrix: ArrayLike | None = None,
        verify: bool = True,
        equilibrium_tolerance: float = 1e-8,
    ) -> FeedbackResult:
        x_eq = self.model.as_state(equilibrium, name="equilibrium")
        scale = self.model.as_state(scales, name="scales")
        if np.any(scale <= 0):
            raise ValueError("scales must be strictly positive")

        residual = self.model.evaluate_tendency(x_eq)
        if verify:
            verify_equilibrium(self.model, x_eq, tolerance=equilibrium_tolerance)

        if jacobian_matrix is None:
            matrix = jacobian(
                self.model.tendency,
                x_eq,
                method=derivative_method,
                rel_step=rel_step,
                abs_step=abs_step,
            )
        else:
            matrix = np.asarray(jacobian_matrix, dtype=float)
        expected = (self.model.dimension, self.model.dimension)
        if matrix.shape != expected:
            raise ValueError(f"Jacobian must have shape {expected}, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Jacobian must contain only finite values")

        # Brubaker & Entekhabi (1996), Eq. (19)-(20):
        # A_ij = dG_i/dx_j * sigma_j.
        scaled = matrix * scale[np.newaxis, :]

        process_jacobians: dict[str, FloatArray] = {}
        process_scaled: dict[str, FloatArray] = {}
        for name, function in self.model.processes.items():
            process_matrix = jacobian(
                function,
                x_eq,
                method=derivative_method,
                rel_step=rel_step,
                abs_step=abs_step,
            )
            process_jacobians[name] = process_matrix
            process_scaled[name] = process_matrix * scale[np.newaxis, :]

        diffusion_at_eq = None
        diffusion_matrix = None
        diffusion_scaled = None
        if self.model.diffusion is not None:
            diffusion_at_eq = self.model.evaluate_diffusion(x_eq)
            diffusion_matrix = diffusion_jacobian(
                self.model.diffusion,
                x_eq,
                method=derivative_method,
                rel_step=rel_step,
                abs_step=abs_step,
            )
            # Eq. (32): Lambda_ij = dg_i/dx_j * sigma_j.
            diffusion_scaled = diffusion_matrix * scale

        return FeedbackResult(
            model=self.model,
            equilibrium=x_eq,
            scales=scale,
            jacobian=matrix,
            scaled_matrix=scaled,
            equilibrium_residual=residual,
            process_jacobians=process_jacobians,
            process_scaled_matrices=process_scaled,
            diffusion_at_equilibrium=diffusion_at_eq,
            diffusion_jacobian=diffusion_matrix,
            diffusion_scaled_sensitivity=diffusion_scaled,
            derivative_method=derivative_method,
        )

    def at_equilibrium(
        self,
        *,
        x0: ArrayLike,
        scales: ArrayLike,
        bounds: tuple[ArrayLike, ArrayLike] | None = None,
        solve_tolerance: float = 1e-10,
        max_evaluations: int = 2000,
        **analysis_kwargs,
    ) -> FeedbackResult:
        solution = solve_equilibrium(
            self.model,
            x0,
            bounds=bounds,
            tolerance=solve_tolerance,
            max_evaluations=max_evaluations,
        )
        if not solution.success:
            raise RuntimeError(f"equilibrium solve failed: {solution.message}")
        return self.analyze(
            equilibrium=solution.state,
            scales=scales,
            equilibrium_tolerance=max(solve_tolerance * 10.0, 1e-8),
            **analysis_kwargs,
        )
