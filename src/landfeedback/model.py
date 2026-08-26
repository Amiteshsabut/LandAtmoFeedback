"""Model interfaces used by the feedback diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
VectorFunction = Callable[[FloatArray], ArrayLike]
DiffusionFunction = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class CoupledModel:
    """A coupled dynamical model and optional process decomposition.

    Parameters
    ----------
    state_names:
        Unique names of the prognostic states.
    tendency:
        Function ``G(x)`` returning one deterministic tendency per state.
    diffusion:
        Optional function ``g(x)``. It may return shape ``(n_states,)`` for
        one shared Wiener process or ``(n_states, n_forcings)`` for multiple
        independent Wiener processes.
    processes:
        Optional named component tendency functions. Their sum should reproduce
        ``tendency`` when a complete process budget is supplied.
    units:
        Optional unit labels for the states.
    time_unit:
        Label for the model time unit, for example ``"day"``.
    """

    state_names: Sequence[str]
    tendency: VectorFunction
    diffusion: DiffusionFunction | None = None
    processes: Mapping[str, VectorFunction] = field(default_factory=dict)
    units: Sequence[str] | None = None
    time_unit: str = "time unit"

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.state_names)
        if not names or any(not name.strip() for name in names):
            raise ValueError("state_names must contain at least one non-empty name")
        if len(names) != len(set(names)):
            raise ValueError("state_names must be unique")
        if not callable(self.tendency):
            raise TypeError("tendency must be callable")
        if self.diffusion is not None and not callable(self.diffusion):
            raise TypeError("diffusion must be callable")

        process_map = dict(self.processes)
        if any(not str(name).strip() for name in process_map):
            raise ValueError("process names must be non-empty")
        if any(not callable(function) for function in process_map.values()):
            raise TypeError("every process tendency must be callable")

        if self.units is None:
            units = tuple("" for _ in names)
        else:
            units = tuple(str(unit) for unit in self.units)
            if len(units) != len(names):
                raise ValueError("units must have one entry per state")

        if not str(self.time_unit).strip():
            raise ValueError("time_unit must be non-empty")

        object.__setattr__(self, "state_names", names)
        object.__setattr__(self, "processes", process_map)
        object.__setattr__(self, "units", units)

    @property
    def dimension(self) -> int:
        """Number of prognostic states."""

        return len(self.state_names)

    def state_index(self, state: str | int) -> int:
        """Resolve a state name or integer index."""

        if isinstance(state, str):
            try:
                return self.state_names.index(state)
            except ValueError as exc:
                raise KeyError(f"unknown state {state!r}") from exc
        index = int(state)
        if not 0 <= index < self.dimension:
            raise IndexError(f"state index {index} is outside [0, {self.dimension})")
        return index

    def as_state(self, value: ArrayLike, *, name: str = "state") -> FloatArray:
        """Validate and return a one-dimensional finite state vector."""

        array = np.asarray(value, dtype=float)
        if array.shape != (self.dimension,):
            raise ValueError(f"{name} must have shape ({self.dimension},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def evaluate_tendency(self, state: ArrayLike) -> FloatArray:
        """Evaluate and validate the deterministic tendency."""

        x = self.as_state(state)
        value = np.asarray(self.tendency(x), dtype=float)
        if value.shape != (self.dimension,):
            raise ValueError(
                f"tendency must return shape ({self.dimension},), got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("tendency returned non-finite values")
        return value

    def evaluate_diffusion(self, state: ArrayLike) -> FloatArray:
        """Evaluate and validate the optional diffusion function."""

        if self.diffusion is None:
            raise ValueError("this model does not define a diffusion function")
        x = self.as_state(state)
        value = np.asarray(self.diffusion(x), dtype=float)
        if value.ndim not in (1, 2) or value.shape[0] != self.dimension:
            raise ValueError(
                "diffusion must return shape (n_states,) or (n_states, n_forcings)"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("diffusion returned non-finite values")
        return value

    def evaluate_processes(self, state: ArrayLike) -> dict[str, FloatArray]:
        """Evaluate and validate all named process tendencies."""

        x = self.as_state(state)
        result: dict[str, FloatArray] = {}
        for name, function in self.processes.items():
            value = np.asarray(function(x), dtype=float)
            if value.shape != (self.dimension,):
                raise ValueError(
                    f"process {name!r} must return shape ({self.dimension},), "
                    f"got {value.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"process {name!r} returned non-finite values")
            result[name] = value
        return result

