"""Optional publication-oriented plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

    from .feedback import FeedbackResult


def plot_feedback_matrix(
    result: "FeedbackResult",
    *,
    ax: "Axes | None" = None,
    cmap: str = "RdBu_r",
    annotate: bool = True,
    colorbar: bool = True,
) -> "Axes":
    """Plot a scaled feedback matrix."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))
    limit = float(np.max(np.abs(result.scaled_matrix)))
    limit = limit if limit > 0 else 1.0
    image = ax.imshow(result.scaled_matrix, cmap=cmap, vmin=-limit, vmax=limit)
    names = result.model.state_names
    ax.set_xticks(range(len(names)), labels=names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_xlabel("Predictor anomaly")
    ax.set_ylabel("Target tendency")
    if annotate:
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{result.scaled_matrix[i, j]:.2g}", ha="center", va="center")
    if colorbar:
        ax.figure.colorbar(image, ax=ax, label="Tendency per 1-SD anomaly")
    return ax


def plot_contributions(
    result: "FeedbackResult",
    state: ArrayLike,
    *,
    target: str | int,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot pathway contributions for one target at a specified state."""

    import matplotlib.pyplot as plt

    index = result.model.state_index(target)
    values = result.contributions(state)[index]
    classes = result.classifications(state)[index]
    colors = [
        "#2c7bb6" if label == "restoring" else "#d7191c" if label == "reinforcing" else "#999999"
        for label in classes
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(result.model.state_names, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Contribution to {result.model.state_names[index]} tendency")
    ax.tick_params(axis="x", rotation=35)
    return ax

