"""
FedMed Differential Privacy Utilities

Provides clipping and Gaussian noise mechanisms for
protecting local model updates before federated aggregation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def clip_update(
    update: Dict[str, np.ndarray],
    max_norm: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Clip a model update to a maximum global L2 norm.

    Non-floating-point arrays are preserved unchanged.
    """

    if max_norm <= 0:
        raise ValueError("max_norm must be greater than zero.")

    total_squared_norm = 0.0

    for value in update.values():
        array = np.asarray(value)

        if np.issubdtype(array.dtype, np.floating):
            total_squared_norm += float(np.sum(array.astype(np.float64) ** 2))

    total_norm = float(np.sqrt(total_squared_norm))

    if total_norm <= max_norm:
        return {
            key: np.asarray(value).copy()
            for key, value in update.items()
        }

    scale = max_norm / (total_norm + 1e-12)

    clipped = {}

    for key, value in update.items():
        array = np.asarray(value)

        if np.issubdtype(array.dtype, np.floating):
            clipped[key] = array * scale
        else:
            clipped[key] = array.copy()

    return clipped


def add_gaussian_noise(
    update: Dict[str, np.ndarray],
    noise_multiplier: float = 0.1,
    max_norm: float = 1.0,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Clip a model update and add Gaussian noise to
    floating-point parameters.

    Non-floating-point arrays are preserved unchanged.
    """

    if noise_multiplier < 0:
        raise ValueError("noise_multiplier cannot be negative.")

    if max_norm <= 0:
        raise ValueError("max_norm must be greater than zero.")

    clipped = clip_update(
        update,
        max_norm=max_norm,
    )

    rng = np.random.default_rng(seed)

    protected = {}

    for key, value in clipped.items():

        array = np.asarray(value)

        if np.issubdtype(array.dtype, np.floating):

            noise = rng.normal(
                loc=0.0,
                scale=noise_multiplier * max_norm,
                size=array.shape,
            )

            protected[key] = (
                array.astype(np.float64) + noise
            ).astype(array.dtype)

        else:
            protected[key] = array.copy()

    return protected