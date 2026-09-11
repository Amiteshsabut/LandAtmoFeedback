import numpy as np


def test_scaled_matrix_matches_paper_convention(linear_case):
    result = linear_case["result"]
    expected = linear_case["jacobian"] * linear_case["scales"][None, :]
    np.testing.assert_allclose(result.jacobian, linear_case["jacobian"], atol=1e-9)
    np.testing.assert_allclose(result.scaled_matrix, expected, atol=1e-9)
    np.testing.assert_allclose(result.sigma_scaled_matrix, expected, atol=1e-9)


def test_contributions_sum_to_linearized_tendency(linear_case):
    result = linear_case["result"]
    state = linear_case["equilibrium"] + linear_case["scales"] * np.array([1.0, -1.0])
    np.testing.assert_allclose(
        result.contributions(state).sum(axis=1),
        result.linearized_tendency(state),
        atol=1e-12,
    )


def test_contributions_and_classification(linear_case):
    result = linear_case["result"]
    state = linear_case["equilibrium"] + linear_case["scales"] * np.array([1.0, -1.0])
    expected = np.array([[-0.5, -0.5], [-0.05, 1.0]])
    np.testing.assert_allclose(result.contributions(state), expected)
    assert result.classifications(state).tolist() == [
        ["restoring", "restoring"],
        ["reinforcing", "restoring"],
    ]


def test_process_closure_and_diffusion(linear_case):
    result = linear_case["result"]
    np.testing.assert_allclose(result.process_closure(), 0.0, atol=1e-9)
    np.testing.assert_allclose(
        result.diffusion_jacobian,
        [[0.1, 0.0], [0.0, -0.05]],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.diffusion_scaled_sensitivity,
        [[0.05, 0.0], [0.0, -0.1]],
        atol=1e-9,
    )


def test_diffusion_linearization_exact_for_linear_diffusion(linear_case):
    result = linear_case["result"]
    state = np.array([1.2, 1.6])
    np.testing.assert_allclose(
        result.linearized_diffusion(state),
        result.model.evaluate_diffusion(state),
        atol=1e-12,
    )


def test_standardized_coordinate_eigenvalues_are_invariant(linear_case):
    result = linear_case["result"]
    raw = np.sort_complex(np.linalg.eigvals(result.jacobian))
    std = np.sort_complex(np.linalg.eigvals(result.standardized_dynamics_matrix))
    np.testing.assert_allclose(raw, std, atol=1e-12)
