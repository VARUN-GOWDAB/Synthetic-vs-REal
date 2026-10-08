from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def save_metric_curve(metric_values: Sequence[float], labels: Sequence[str], output_path: str | Path, title: str = "Metric curve") -> None:
    """Write a simple text-based metric plot when matplotlib is unavailable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(metric_values)), metric_values, marker="o")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
