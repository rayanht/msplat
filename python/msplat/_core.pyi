"""Type stubs for msplat._core (compiled nanobind extension)."""

import numpy as np
from numpy.typing import NDArray

class TrainingConfig:
    iterations: int
    sh_degree: int
    sh_degree_interval: int
    ssim_weight: float
    num_downscales: int
    resolution_schedule: int
    refine_every: int
    warmup_length: int
    reset_alpha_every: int
    densify_grad_thresh: float
    densify_size_thresh: float
    stop_screen_size_at: int
    split_screen_size: float
    keep_crs: bool
    downscale_factor: float
    output: str
    save_every: int
    bg_color: list[float]
    """Background color as [R, G, B] floats in [0, 1]. Default magenta [0.613, 0.010, 0.398]."""

    def __init__(
        self,
        iterations: int = 30000,
        sh_degree: int = 3,
        sh_degree_interval: int = 1000,
        ssim_weight: float = 0.2,
        num_downscales: int = 2,
        resolution_schedule: int = 3000,
        refine_every: int = 100,
        warmup_length: int = 500,
        reset_alpha_every: int = 30,
        densify_grad_thresh: float = 0.0002,
        densify_size_thresh: float = 0.01,
        stop_screen_size_at: int = 4000,
        split_screen_size: float = 0.05,
        keep_crs: bool = False,
        downscale_factor: float = 1.0,
        output: str = "splat.ply",
        save_every: int = -1,
        bg_color: list[float] = ...,
    ) -> None: ...

class TrainingStats:
    """Per-step training statistics returned by GaussianTrainer.step()."""

    @property
    def iteration(self) -> int:
        """Current training iteration."""
        ...

    @property
    def splat_count(self) -> int:
        """Number of active Gaussians."""
        ...

    @property
    def ms_per_step(self) -> float:
        """Wall-clock time for this step in milliseconds."""
        ...

class Dataset:
    """A loaded dataset of camera images. Auto-detects COLMAP, Nerfstudio, and Polycam formats."""

    def __init__(
        self,
        path: str,
        downscale_factor: float = 1.0,
        eval_mode: bool = False,
        test_every: int = 8,
    ) -> None: ...

    @property
    def num_train(self) -> int:
        """Number of training cameras."""
        ...

    @property
    def num_test(self) -> int:
        """Number of test cameras (0 unless eval_mode=True)."""
        ...

class GaussianTrainer:
    """3D Gaussian Splatting trainer. All computation runs on the Metal GPU."""

    def __init__(self, dataset: Dataset, config: TrainingConfig) -> None: ...

    def step(self) -> TrainingStats:
        """Run a single training iteration. Returns TrainingStats."""
        ...

    def train(
        self,
        callback: object,
        callback_every: int = 100,
    ) -> None:
        """Run training to completion, calling callback(stats) every callback_every steps."""
        ...

    def evaluate(self) -> dict[str, float | int]:
        """Evaluate on held-out test cameras. Returns dict with psnr, ssim, l1 keys.

        Requires the dataset to have been loaded with eval_mode=True.
        """
        ...

    def render(
        self,
        cam_idx: int,
        use_test: bool = False,
    ) -> NDArray[np.float32]:
        """Render a camera view. Returns a numpy array of shape (H, W, 3), float32, RGB [0,1]."""
        ...

    def export_ply(self, path: str) -> None:
        """Export the current Gaussians as a PLY file."""
        ...

    def export_splat(self, path: str) -> None:
        """Export the current Gaussians as a .splat file."""
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save a training checkpoint."""
        ...

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint and resume from the saved iteration."""
        ...

    @property
    def splat_count(self) -> int:
        """Current number of active Gaussians."""
        ...

    @property
    def iteration(self) -> int:
        """Current training iteration."""
        ...

def sync() -> None:
    """Synchronize GPU (wait for all commands to complete)."""
    ...

def cleanup() -> None:
    """Release all cached GPU resources."""
    ...
