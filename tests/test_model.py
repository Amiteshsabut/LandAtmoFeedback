from __future__ import annotations

import numpy as np
import pytest

from landfeedback import CoupledModel


def test_model_metadata_and_evaluation():
    model = CoupledModel(
        state_names=["a", "b"],
        tendency=lambda x: -x,
        units=["m", "K"],
    )
    assert model.dimension == 2
    assert model.state_index("b") == 1
    np.testing.assert_allclose(model.evaluate_tendency([1, 2]), [-1, -2])


@pytest.mark.parametrize(
    "names",
    [[], [""], ["a", "a"]],
)
def test_invalid_state_names(names):
    with pytest.raises(ValueError):
        CoupledModel(state_names=names, tendency=lambda x: x)


def test_invalid_units_and_shapes():
    with pytest.raises(ValueError, match="units"):
        CoupledModel(state_names=["a", "b"], tendency=lambda x: x, units=["m"])

    model = CoupledModel(state_names=["a", "b"], tendency=lambda x: x)
    with pytest.raises(ValueError, match="shape"):
        model.as_state([1])
    with pytest.raises(ValueError, match="shape"):
        model.evaluate_tendency([1, 2, 3])


def test_invalid_tendency_output():
    model = CoupledModel(state_names=["a", "b"], tendency=lambda x: [1])
    with pytest.raises(ValueError, match="tendency"):
        model.evaluate_tendency([1, 2])


def test_diffusion_validation():
    vector = CoupledModel(
        state_names=["a", "b"],
        tendency=lambda x: -x,
        diffusion=lambda x: [1.0, 2.0],
    )
    assert vector.evaluate_diffusion([0, 0]).shape == (2,)

    matrix = CoupledModel(
        state_names=["a", "b"],
        tendency=lambda x: -x,
        diffusion=lambda x: [[1.0, 0.0], [0.0, 2.0]],
    )
    assert matrix.evaluate_diffusion([0, 0]).shape == (2, 2)

    missing = CoupledModel(state_names=["a"], tendency=lambda x: -x)
    with pytest.raises(ValueError, match="does not define"):
        missing.evaluate_diffusion([0])

