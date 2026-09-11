"""Tools for diagnosing coupled land-atmosphere feedbacks."""

from .conditional import ConditionalComparison, conditional_composites
from .decomposition import ReportingDecomposition
from .derivatives import derivative_convergence, diffusion_jacobian, jacobian
from .equilibrium import EquilibriumSolution, solve_equilibrium, verify_equilibrium
from .feedback import FeedbackAnalyzer, FeedbackResult
from .model import CoupledModel
from .stochastic import SimulationResult, simulate_sde
from .validation import ValidationReport, full_validation_report

__all__ = [
    "ConditionalComparison",
    "CoupledModel",
    "EquilibriumSolution",
    "FeedbackAnalyzer",
    "FeedbackResult",
    "ReportingDecomposition",
    "SimulationResult",
    "ValidationReport",
    "conditional_composites",
    "derivative_convergence",
    "diffusion_jacobian",
    "full_validation_report",
    "jacobian",
    "simulate_sde",
    "solve_equilibrium",
    "verify_equilibrium",
]

__version__ = "0.2.0"
