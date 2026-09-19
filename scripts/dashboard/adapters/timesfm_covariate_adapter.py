"""Stub adapter for `timesfm_covariate` — fills in once a checkpoint exists."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from scripts.lib.inference import ModelNotAvailableError, ModelInput

ForwardFn = Callable[[ModelInput], "np.ndarray"]  # noqa: F821


def load_checkpoint(checkpoint: Path) -> ForwardFn:
    raise ModelNotAvailableError(
        "timesfm_covariate 尚未訓練完成（Phase 3 進行中）。"
    )
