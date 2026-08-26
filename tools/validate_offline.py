"""Standard-library validation for environments without pytest or ruff."""

from __future__ import annotations

import unittest

import numpy as np

from landfeedback import (
    CoupledModel,
    FeedbackAnalyzer,
    jacobian,
    simulate_sde,
    solve_equilibrium,
)
from landfeedback.benchmarks import brubaker1996


class OfflineValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.equilibrium = np.array([1.0, 2.0])
        self.scales = np.array([0.5, 2.0])
        self.p1_matrix = np.array([[-1.0, 0.0], [-0.1, 0.0]])
        self.p2_matrix = np.array([[0.0, 0.25], [0.0, -0.5]])

        def p1(x):
            return self.p1_matrix @ (x - self.equilibrium)

        def p2(x):
            return self.p2_matrix @ (x - self.equilibrium)

        def diffusion(x):
            return np.array([0.2 + 0.1 * (x[0] - 1), 0.3 - 0.05 * (x[1] - 2)])

        self.model = CoupledModel(
            state_names=["soil_moisture", "soil_temperature"],
            tendency=lambda x: p1(x) + p2(x),
            diffusion=diffusion,
            processes={"water": p1, "energy": p2},
            time_unit="day",
        )
        self.result = FeedbackAnalyzer(self.model).analyze(
            equilibrium=self.equilibrium,
            scales=self.scales,
        )

    def test_derivatives_and_feedbacks(self) -> None:
        expected_jacobian = self.p1_matrix + self.p2_matrix
        np.testing.assert_allclose(self.result.jacobian, expected_jacobian, atol=1e-9)
        np.testing.assert_allclose(self.result.process_closure(), 0.0, atol=1e-9)
        state = self.equilibrium + self.scales * np.array([1.0, -1.0])
        self.assertEqual(
            self.result.classifications(state).tolist(),
            [["restoring", "restoring"], ["reinforcing", "restoring"]],
        )
        self.assertTrue(self.result.locally_stable)

    def test_complex_step(self) -> None:
        function = lambda x: np.array([x[0] ** 2 + x[1], x[0] * x[1]])
        expected = np.array([[4.0, 1.0], [3.0, 2.0]])
        np.testing.assert_allclose(
            jacobian(function, [2.0, 3.0], method="complex"), expected, atol=1e-10
        )

    def test_equilibrium_and_conditionals(self) -> None:
        solution = solve_equilibrium(self.model, [0.0, 0.0])
        self.assertTrue(solution.success)
        np.testing.assert_allclose(solution.state, self.equilibrium, atol=1e-9)
        values = np.linspace(-2.0, 2.0, 101)
        standardized = np.column_stack([values, -0.5 * values])
        states = self.equilibrium + standardized * self.scales
        comparison = self.result.conditional_composites(
            states,
            conditioning="soil_moisture",
            lower_quantile=0.1,
            upper_quantile=0.9,
        )
        self.assertEqual(comparison.lower_count, 11)
        self.assertEqual(comparison.upper_count, 11)
        table = comparison.tendency_frame()
        np.testing.assert_allclose(table.lower_linear, table.lower_nonlinear, atol=1e-10)
        np.testing.assert_allclose(table.upper_linear, table.upper_nonlinear, atol=1e-10)

    def test_stochastic_simulation(self) -> None:
        first = simulate_sde(self.model, self.equilibrium, dt=0.01, n_steps=20, seed=42)
        second = simulate_sde(self.model, self.equilibrium, dt=0.01, n_steps=20, seed=42)
        self.assertEqual(first.states.shape, (1, 21, 2))
        np.testing.assert_allclose(first.states, second.states)

    def test_published_benchmark(self) -> None:
        matrix = brubaker1996.published_reporting_matrix()
        self.assertEqual(matrix.shape, (2, 5))
        self.assertEqual(matrix.loc["soil_moisture_tendency", "soil_moisture"], -0.64)
        self.assertEqual(
            matrix.loc["soil_temperature_tendency", "delta_temperature"], -2.91
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

