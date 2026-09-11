"""Conditional dry/wet feedback composites following the 1996 analysis logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .feedback import FeedbackResult

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ConditionalComparison:
    """Lower- and upper-tail composites around one equilibrium linearization.

    This mirrors Brubaker & Entekhabi (1996) section 6: coefficients are
    evaluated at the equilibrium, while physically consistent disequilibria are
    obtained from lower/upper tails of the joint state distribution.
    """

    result: "FeedbackResult"
    conditioning_index: int
    lower_quantile: float
    upper_quantile: float
    lower_threshold: float
    upper_threshold: float
    lower_count: int
    upper_count: int
    lower_mean_state: FloatArray
    upper_mean_state: FloatArray
    lower_contributions: FloatArray
    upper_contributions: FloatArray
    lower_nonlinear_tendency: FloatArray
    upper_nonlinear_tendency: FloatArray

    @property
    def conditioning_state(self) -> str:
        return self.result.model.state_names[self.conditioning_index]

    @property
    def lower_standardized_disequilibrium(self) -> FloatArray:
        return self.result.standardized_anomaly(self.lower_mean_state)

    @property
    def upper_standardized_disequilibrium(self) -> FloatArray:
        return self.result.standardized_anomaly(self.upper_mean_state)

    @property
    def lower_linearized_tendency(self) -> FloatArray:
        return self.lower_contributions.sum(axis=1)

    @property
    def upper_linearized_tendency(self) -> FloatArray:
        return self.upper_contributions.sum(axis=1)

    def contribution_frame(self) -> pd.DataFrame:
        rows = []
        names = self.result.model.state_names
        for i, target in enumerate(names):
            for j, predictor in enumerate(names):
                lower = self.lower_contributions[i, j]
                upper = self.upper_contributions[i, j]
                rows.append(
                    {
                        "target": target,
                        "predictor": predictor,
                        "lower_contribution": lower,
                        "upper_contribution": upper,
                        "upper_minus_lower": upper - lower,
                    }
                )
        return pd.DataFrame(rows)

    def disequilibrium_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "state": self.result.model.state_names,
                "lower_standardized": self.lower_standardized_disequilibrium,
                "upper_standardized": self.upper_standardized_disequilibrium,
            }
        )

    def tendency_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "state": self.result.model.state_names,
                "lower_linear": self.lower_linearized_tendency,
                "lower_nonlinear": self.lower_nonlinear_tendency,
                "upper_linear": self.upper_linearized_tendency,
                "upper_nonlinear": self.upper_nonlinear_tendency,
            }
        )

    def asymmetry_index(self, target: str | int | None = None) -> float:
        if target is None:
            lower = self.lower_contributions.ravel()
            upper = self.upper_contributions.ravel()
        else:
            index = self.result.model.state_index(target)
            lower = self.lower_contributions[index]
            upper = self.upper_contributions[index]
        denominator = np.linalg.norm(upper) + np.linalg.norm(lower)
        if denominator == 0:
            return 0.0
        return float(np.linalg.norm(upper - lower) / denominator)


def conditional_composites(
    result: "FeedbackResult",
    states: ArrayLike,
    *,
    conditioning: str | int,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> ConditionalComparison:
    """Calculate lower/upper tail composites using one equilibrium linearization.

    Defaults reproduce the fifth and 95th percentile soil-moisture selection
    used in Brubaker & Entekhabi (1996).
    """

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    samples = np.asarray(states, dtype=float)
    expected_columns = result.model.dimension
    if samples.ndim != 2 or samples.shape[1] != expected_columns:
        raise ValueError(
            f"states must have shape (n_samples, {expected_columns}), got {samples.shape}"
        )
    if samples.shape[0] < 2:
        raise ValueError("states must contain at least two samples")
    if not np.all(np.isfinite(samples)):
        raise ValueError("states must contain only finite values")

    index = result.model.state_index(conditioning)
    values = samples[:, index]
    lower_threshold = float(np.quantile(values, lower_quantile))
    upper_threshold = float(np.quantile(values, upper_quantile))
    lower_samples = samples[values <= lower_threshold]
    upper_samples = samples[values >= upper_threshold]
    if lower_samples.size == 0 or upper_samples.size == 0:
        raise ValueError("one conditional tail contains no samples")

    lower_mean = lower_samples.mean(axis=0)
    upper_mean = upper_samples.mean(axis=0)
    lower_nonlinear = np.mean(
        [result.model.evaluate_tendency(row) for row in lower_samples], axis=0
    )
    upper_nonlinear = np.mean(
        [result.model.evaluate_tendency(row) for row in upper_samples], axis=0
    )

    return ConditionalComparison(
        result=result,
        conditioning_index=index,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        lower_count=len(lower_samples),
        upper_count=len(upper_samples),
        lower_mean_state=lower_mean,
        upper_mean_state=upper_mean,
        lower_contributions=result.contributions(lower_mean),
        upper_contributions=result.contributions(upper_mean),
        lower_nonlinear_tendency=np.asarray(lower_nonlinear, dtype=float),
        upper_nonlinear_tendency=np.asarray(upper_nonlinear, dtype=float),
    )
