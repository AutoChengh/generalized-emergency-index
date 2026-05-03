"""GEI computation package."""

__all__ = ["compute_single_frame", "process_one_csv"]


def compute_single_frame(*args, **kwargs):
    """Compute GEI and related metrics for one frame."""
    from .cli import compute_single_frame as _compute_single_frame

    return _compute_single_frame(*args, **kwargs)


def process_one_csv(*args, **kwargs):
    """Compute GEI and related metrics for every frame in one CSV file."""
    from .cli import process_one_csv as _process_one_csv

    return _process_one_csv(*args, **kwargs)
