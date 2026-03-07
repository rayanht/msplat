"""msplat: Metal-accelerated 3D Gaussian Splatting."""

import atexit

from msplat._core import (
    TrainingConfig,
    TrainingStats,
    Dataset,
    GaussianTrainer,
    sync,
    cleanup as _cleanup_raw,
)

_cleaned_up = False


def cleanup():
    """Release all cached GPU resources. Safe to call multiple times."""
    global _cleaned_up
    if not _cleaned_up:
        _cleaned_up = True
        _cleanup_raw()


atexit.register(cleanup)

__all__ = [
    "TrainingConfig",
    "TrainingStats",
    "Dataset",
    "GaussianTrainer",
    "sync",
    "cleanup",
    "load_dataset",
]

__version__ = "1.1.3"


def load_dataset(
    path: str,
    downscale_factor: float = 1.0,
    eval_mode: bool = False,
    test_every: int = 8,
) -> Dataset:
    """Load a dataset (auto-detects COLMAP, Nerfstudio, Polycam)."""
    return Dataset(path, downscale_factor, eval_mode, test_every)
