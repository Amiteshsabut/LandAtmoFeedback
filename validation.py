"""Scientific verification and benchmark-validation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .benchmarks import brubaker1996
from .derivatives import derivative_convergence, jacobian
from .feedback import FeedbackAnalyzer
from .model import CoupledModel


@dataclass(frozen=True)
class ValidationReport:
    checks: pd.DataFrame
    derivative_convergence: pd.DataFrame
    paper_arithmetic: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool(self.checks["passed"].all())

    def summary(self) -> str:
        lines = ["LandAtmoFeedback validation report", "=" * 37]
        for row in self.checks.itertuples(index=False):
            mark = "PASS" if row.passed else "FAIL"
            lines.append(f"{mark:4s}  {row.check}")
        lines.append("-" * 37)
        lines.append("OVERALL: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)

    def assert_valid(self) -> None:
        if not self.passed:
            failed = self.checks.loc[~self.checks["passed"], "check"].tolist()
            raise AssertionError("validation failures: " + ", ".join(failed))


def _analytic_case():
    """Nonlinear two-state system with exact derivatives."""

    equilibrium = np.array([0.6, 20.0])
    scales = np.array([0.08, 2.0])

    def water(x):
        s, t = x
        ds = s - equilibrium[0]
        dt = t - equilibrium[1]
        return np.array([
            -0.40 * ds - 0.05 * dt - 0.10 * ds * dt,
            -1.20 * ds,
        ])

    def energy(x):
        s, t = x
        ds = s - equilibrium[0]
        dt = t - equilibrium[1]
        return np.array([
            0.0 * ds,
            -0.80 * dt + 0.20 * ds**2,
        ])

    def tendency(x):
        return water(x) + energy(x)

    def diffusion(x):
        s, t = x
        ds = s - equilibrium[0]
        dt = t - equilibrium[1]
        return np.array([
            0.30 + 0.10 * ds - 0.02 * dt,
            0.50 - 0.20 * ds + 0.03 * dt,
        ])

    exact_j = np.array([
        [-0.40, -0.05],
        [-1.20, -0.80],
    ])
    exact_gj = np.array([
        [0.10, -0.02],
        [-0.20, 0.03],
    ])

    model = CoupledModel(
        state_names=("soil_moisture", "temperature"),
        tendency=tendency,
        diffusion=diffusion,
        processes={"water": water, "energy": energy},
        units=("1", "degC"),
        time_unit="day",
    )
    return model, equilibrium, scales, exact_j, exact_gj


def full_validation_report() -> ValidationReport:
    """Run analytic verification + published Brubaker-1996 regression checks."""

    model, eq, scales, exact_j, exact_gj = _analytic_case()
    analyzer = FeedbackAnalyzer(model)
    result = analyzer.analyze(equilibrium=eq, scales=scales)

    rows: list[dict[str, object]] = []

    def add(check: str, passed: bool, metric: float = 0.0, tolerance: float = 0.0):
        rows.append(
            {
                "check": check,
                "passed": bool(passed),
                "metric": float(metric),
                "tolerance": float(tolerance),
            }
        )

    err_j = float(np.max(np.abs(result.jacobian - exact_j)))
    add("analytic nonlinear Jacobian", err_j < 1e-8, err_j, 1e-8)

    complex_j = jacobian(model.tendency, eq, method="complex", rel_step=1e-20)
    err_complex = float(np.max(np.abs(complex_j - exact_j)))
    add("complex-step Jacobian cross-check", err_complex < 1e-12, err_complex, 1e-12)

    err_diff = float(np.max(np.abs(result.diffusion_jacobian - exact_gj)))
    add("analytic diffusion Jacobian", err_diff < 1e-8, err_diff, 1e-8)

    closure = float(np.max(np.abs(result.process_closure())))
    add("deterministic process closure", closure < 1e-8, closure, 1e-8)

    expected_scaled = exact_j * scales[None, :]
    scaled_err = float(np.max(np.abs(result.scaled_matrix - expected_scaled)))
    add("Eq.20 one-sigma scaling A_ij=J_ij sigma_j", scaled_err < 1e-8, scaled_err, 1e-8)

    eig_physical = np.sort_complex(np.linalg.eigvals(result.jacobian))
    eig_standardized = np.sort_complex(np.linalg.eigvals(result.standardized_dynamics_matrix))
    eig_err = float(np.max(np.abs(eig_physical - eig_standardized)))
    add("D^-1 J D eigenvalue invariance", eig_err < 1e-10, eig_err, 1e-10)

    test_state = eq + scales * np.array([-0.5, 0.7])
    contribution_total = result.contributions(test_state).sum(axis=1)
    direct_linear = result.linearized_tendency(test_state)
    contrib_err = float(np.max(np.abs(contribution_total - direct_linear)))
    add("Eq.20 contribution sum equals J(x-x*)", contrib_err < 1e-10, contrib_err, 1e-10)

    # Local Taylor error should contract rapidly when perturbation size is halved.
    direction = scales * np.array([0.8, -0.6])
    errors = []
    for factor in (0.2, 0.1, 0.05):
        state = eq + factor * direction
        errors.append(float(np.linalg.norm(result.linearization_error(state))))
    taylor_ok = errors[2] < errors[1] < errors[0]
    add("nonlinear Taylor error decreases near equilibrium", taylor_ok, errors[-1], errors[0])

    g_state = eq + scales * np.array([0.4, -0.3])
    g_err = float(np.max(np.abs(result.diffusion_linearization_error(g_state))))
    add("Eq.32 diffusion linearization", g_err < 1e-10, g_err, 1e-10)

    # Published paper arithmetic: allow 0.03 because source tables are rounded to 0.01.
    paper = brubaker1996.paper_arithmetic_checks()
    max_paper = float(paper["absolute_difference"].max())
    add("Brubaker 1996 Tables 4-7 arithmetic", max_paper <= 0.03, max_paper, 0.03)

    # Explicit published-value spot checks from Tables 3,4,5,8,9.
    states = brubaker1996.state_statistics().set_index("variable")
    fixture_ok = (
        np.isclose(states.loc["soil_moisture", "equilibrium"], 0.613)
        and np.isclose(states.loc["soil_temperature", "stationary_sd"], 2.1)
        and np.isclose(
            brubaker1996.published_reporting_matrix().loc[
                "soil_moisture_tendency", "soil_moisture"
            ],
            -0.64,
        )
        and np.isclose(
            brubaker1996.stochastic_soil_moisture()
            .set_index("term")
            .loc["g1_equilibrium", "scaled_value_mm_day"],
            0.52,
        )
        and np.isclose(
            brubaker1996.stochastic_soil_temperature()
            .set_index("term")
            .loc["Lambda33", "scaled_value_degC_day"],
            -0.53,
        )
    )
    add("published-value regression fixtures", fixture_ok)

    convergence = derivative_convergence(
        model.tendency,
        eq,
        reference=exact_j,
        methods=("central", "forward", "complex"),
    )

    checks = pd.DataFrame(rows)
    return ValidationReport(
        checks=checks,
        derivative_convergence=convergence,
        paper_arithmetic=paper,
    )
