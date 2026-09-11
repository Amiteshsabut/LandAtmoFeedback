"""Paper-style reporting decompositions.

This module supports coefficient tables whose reporting predictors may include
derived channels.  Brubaker & Entekhabi (1996), for example, report
``delta_temperature = soil_temperature - air_temperature`` as a fifth
disequilibrium channel even though the prognostic model has four states.
Such a table is a reporting decomposition, not an independent-state Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ReportingDecomposition:
    target_names: tuple[str, ...]
    predictor_names: tuple[str, ...]
    coefficients: FloatArray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.coefficients, dtype=float)
        expected = (len(self.target_names), len(self.predictor_names))
        if matrix.shape != expected:
            raise ValueError(f"coefficients must have shape {expected}, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("coefficients must be finite")
        object.__setattr__(self, "coefficients", matrix)

    def coefficient_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.coefficients,
            index=pd.Index(self.target_names, name="target"),
            columns=pd.Index(self.predictor_names, name="reporting_predictor"),
        )

    def contributions(self, standardized_disequilibria: ArrayLike) -> FloatArray:
        delta = np.asarray(standardized_disequilibria, dtype=float)
        if delta.shape != (len(self.predictor_names),):
            raise ValueError(
                "standardized_disequilibria must have one value per reporting predictor"
            )
        return self.coefficients * delta[np.newaxis, :]

    def total_tendency(self, standardized_disequilibria: ArrayLike) -> FloatArray:
        return self.contributions(standardized_disequilibria).sum(axis=1)

    def classify(
        self,
        standardized_disequilibria: ArrayLike,
        target_standardized_anomalies: ArrayLike,
        *,
        tolerance: float = 1e-12,
    ) -> NDArray:
        target = np.asarray(target_standardized_anomalies, dtype=float)
        if target.shape != (len(self.target_names),):
            raise ValueError("target_standardized_anomalies must have one value per target")
        signed = self.contributions(standardized_disequilibria) * target[:, None]
        labels = np.full(signed.shape, "neutral", dtype=object)
        labels[signed < -tolerance] = "restoring"
        labels[signed > tolerance] = "reinforcing"
        return labels
