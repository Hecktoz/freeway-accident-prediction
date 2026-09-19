"""Adapter for ``unet_feature_fusion`` checkpoints via ``inference.load``."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable

from scripts.lib.inference import ModelInput

ForwardFn = Callable[[ModelInput], np.ndarray]


def load_checkpoint(checkpoint: Path) -> ForwardFn:
    """Load a feature-fusion checkpoint and return a predict_from_input callable.

    Expects checkpoint at:
      .../unet_feature_fusion/<run_id>/<experiment>/checkpoints/best_model.pth
    """
    from scripts.lib.inference import load

    # Path: arch_dir / run_id / experiment / checkpoints / best_model.pth
    experiment = checkpoint.parent.parent.name
    run_id = checkpoint.parent.parent.parent.name
    models_root = checkpoint.parent.parent.parent.parent.parent

    model = load(
        "unet_feature_fusion",
        run_id=run_id,
        models_root=models_root,
        experiment=experiment,
    )
    return model.predict_from_input
