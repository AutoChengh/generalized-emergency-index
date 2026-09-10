"""Generalized Emergency Index: one risk functional, multiple future generators."""

from ._solver import Trajectory, TrajectoryEIResult, compute_trajectory_ei
from .api import FutureEvaluation, GEIResult, compute_gei, compute_gei_from_futures
from .futures import JointFuture, independent_joint_futures
from .models import ActorState, BodySize, NumericalError, SolverOptions

__version__ = "0.2.0.dev0"
__all__ = [
    "ActorState",
    "BodySize",
    "Trajectory",
    "JointFuture",
    "SolverOptions",
    "NumericalError",
    "TrajectoryEIResult",
    "FutureEvaluation",
    "GEIResult",
    "compute_gei",
    "compute_gei_from_futures",
    "compute_trajectory_ei",
    "independent_joint_futures",
]
