"""Tools for diagnosing coupled land-atmosphere feedbacks."""

from .conditional import ConditionalComparison, conditional_composites
from .derivatives import diffusion_jacobian, jacobian
from .equilibrium import EquilibriumSolution, solve_equilibrium, verify_equilibrium
from .feedback import FeedbackAnalyzer, FeedbackResult
from .model import CoupledModel
from .stochastic import SimulationResult, simulate_sde

__all__ = [
    "ConditionalComparison",
    "CoupledModel",
    "EquilibriumSolution",
    "FeedbackAnalyzer",
    "FeedbackResult",
    "SimulationResult",
    "conditional_composites",
    "diffusion_jacobian",
    "jacobian",
    "simulate_sde",
    "solve_equilibrium",
    "verify_equilibrium",
]

__version__ = "0.1.0"

