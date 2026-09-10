"""Validated inputs. Distances are metres, times seconds, angles radians."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BodySize:
    """Oriented-rectangle size, centred at the supplied position."""

    length: float
    width: float

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.length, self.width)):
            raise ValueError("body length and width must be finite and positive")

    @property
    def radius(self):
        return 0.5 * math.hypot(self.length, self.width)


@dataclass(frozen=True)
class ActorState:
    """Current centre state; heading and yaw rate are counterclockwise."""

    x: float
    y: float
    speed: float
    heading: float
    yaw_rate: float
    length: float
    width: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("actor state values must be finite")
        if self.speed < 0:
            raise ValueError("speed must be non-negative")
        BodySize(self.length, self.width)

    @property
    def size(self):
        return BodySize(self.length, self.width)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float


@dataclass(frozen=True)
class SolverOptions:
    """Numerical controls, not parameters fitted to crash labels.

    ``safety_margin`` adds to projected intrusion only; it does not inflate
    bodies or change contact occurrence. ``contact_tolerance`` is in m;
    time controls are in s. Accuracy concerns the supplied continuous paths,
    not the accuracy of their predicted futures.
    """

    contact_tolerance: float = 1e-10
    min_search_step: float = 1e-4
    time_tolerance: float = 1e-9
    indepth_max_step: float = 0.01
    safety_margin: float = 0.0
    max_contact_evaluations: int = 250_000

    def __post_init__(self):
        for name in ("contact_tolerance", "safety_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("min_search_step", "time_tolerance", "indepth_max_step"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.time_tolerance > self.min_search_step:
            raise ValueError("time_tolerance must not exceed min_search_step")
        if type(self.max_contact_evaluations) is not int or self.max_contact_evaluations < 10:
            raise ValueError("max_contact_evaluations must be an integer >= 10")


class NumericalError(RuntimeError):
    """Numerical evaluation failed; this is not zero risk."""
