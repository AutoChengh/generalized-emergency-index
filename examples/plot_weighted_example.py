"""Generate the README illustration from the runnable weighted example."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from gei.cli import evaluate_document


def main():
    root = Path(__file__).resolve().parents[1]
    result = evaluate_document(json.loads((root / "examples/weighted.json").read_text()))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "svg.fonttype": "none",
            "svg.hashsalt": "gei-weighted-example",
        }
    )
    fig, (ax, bar) = plt.subplots(
        1, 2, figsize=(10, 3.4), layout="constrained", gridspec_kw={"width_ratios": [1.4, 1]}
    )
    ax.add_patch(Rectangle((-1, -1), 2, 2, facecolor="#35618f", alpha=0.8))
    ax.add_patch(Rectangle((4, -1), 2, 2, facecolor="#cf6243", alpha=0.8))
    ax.annotate("", (3, 0), (0, 0), arrowprops={"arrowstyle": "->", "color": "#35618f", "lw": 2})
    ax.add_patch(Rectangle((2, -1), 2, 2, fill=False, edgecolor="#35618f", linestyle="--"))
    ax.text(0, 1.2, "A now", ha="center")
    ax.text(3, -1.5, "A at 3 s", ha="center")
    ax.text(5, 1.2, "B (stationary)", ha="center")
    ax.set(
        xlim=(-1.5, 6.5),
        ylim=(-2, 2),
        xlabel="Longitudinal position (m)",
        ylabel="Lateral position (m)",
        title="Contacting future: weight 0.25",
    )
    ax.set_aspect("equal")
    values = [result.conditional_ei, 0.0, result.gei]
    bars = bar.bar(
        ["Continue\nEI", "Stop\nEI", "Weighted\nGEI"],
        values,
        color=["#35618f", "#999999", "#38896e"],
        width=0.55,
    )
    bar.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=6)
    bar.set(ylim=(0, 0.85), ylabel="Risk (m/s)", title="Same solver, weighted futures")
    bar.spines[["top", "right"]].set_visible(False)
    out = root / "docs/assets"
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "weighted_example.svg"
    fig.savefig(destination, metadata={"Date": None})
    # Keep generated vector text stable and free of trailing whitespace.
    content = destination.read_text(encoding="utf-8")
    destination.write_text(
        "\n".join(line.rstrip() for line in content.splitlines()) + "\n", encoding="utf-8"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
