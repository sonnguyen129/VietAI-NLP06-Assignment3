"""
analysis.py — Post-hoc analysis and visualisation for EvoAgent results.

Three public functions:
  - plot_learning_curve(history):   accuracy vs. iteration line plot.
  - compute_strategy_diversity(history): pairwise embedding distance matrix.
  - failure_mode_report(history):   per-type accuracy trends + a text report.

All plots are saved to PDF. All functions accept an optional output_dir;
if omitted they save to the current working directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Matplotlib is imported lazily inside each function so the module can be
# imported in headless environments without triggering a display backend error.


def _load_eval_accuracy_by_type(output_dir: Path, iteration: int, split: str) -> dict[str, float]:
    """Load per-type accuracy from iter_XXX_eval_train/dev.json when available."""
    path = output_dir / f"iter_{iteration:03d}_eval_{split}.json"
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}

    accuracy_by_type = data.get("accuracy_by_type") or {}
    return {
        str(q_type): float(acc)
        for q_type, acc in accuracy_by_type.items()
        if acc is not None
    }


def plot_learning_curve(
    history,  # StrategyHistory — avoid circular import
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Plot dev (and train) accuracy across EvoAgent iterations.

    Saves a PDF to output_dir/learning_curve.pdf and returns the path.
    Each point corresponds to one evaluated strategy. The best strategy is
    highlighted with a star marker.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = history.summary_table()
    iterations = [r["iteration"] for r in rows]
    dev_accs = [r["dev_accuracy"] for r in rows]
    train_accs = [r["train_accuracy"] for r in rows]

    # Filter out un-evaluated rows.
    scored = [(it, dev, tr) for it, dev, tr in zip(iterations, dev_accs, train_accs) if dev is not None]
    if not scored:
        logger.warning("No evaluated strategies to plot.")
        return output_dir / "learning_curve.pdf"

    iters, devs, trains = zip(*scored)

    # Convert to percentages
    devs = 100 * np.array(devs)
    trains = 100 * np.array(trains)

    seed_row = next((row for row in rows if row["iteration"] == 0), rows[0])
    baseline = 100 * seed_row["dev_accuracy"] if seed_row["dev_accuracy"] is not None else devs[0]

    local_min = min(min(devs), min(trains), baseline)
    local_max = max(max(devs), max(trains), baseline)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, trains, marker="^", color="#F15757", label="Train", lw=2.5, mew=3.5, ms=3.5)
    ax.plot(iters, devs, marker="o", label="Dev", color="royalblue", lw=4, mew=4.2, ms=4.2)

    # Star the best point.
    best_idx = int(np.argmax(devs))
    ax.scatter([iters[best_idx]], [devs[best_idx]], marker="*", s=200, color="gold", zorder=5, label=f"Best ({devs[best_idx]:.3f})")

    ax.axhline(baseline, linestyle="-.", color="#F15757", linewidth=2, label=f"Seed ({baseline:.3f})", zorder=0)

    plt.yticks(np.arange(local_min, local_max + 5, 5.0))
    ax.set_xticks(np.arange(min(iters), max(iters) + 1, 1))

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("EvoAgent Learning Curve — Vietnamese Financial QA")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)

    path = output_dir / "learning_curve.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Learning curve saved to %s", path)
    return path


def compute_strategy_diversity(
    history,  # StrategyHistory
    output_dir: Optional[Path] = None,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> tuple[np.ndarray, Path]:
    """
    Compute pairwise cosine similarity between strategy prompt templates.

    Uses sentence-transformers with a multilingual model so Vietnamese text
    in the templates is handled correctly.

    Returns (similarity_matrix, pdf_path).
    The similarity matrix is (N x N) float32 where N = number of strategies.
    The heatmap is saved to output_dir/strategy_diversity.pdf.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers is not installed. Run: pip install sentence-transformers")
        raise

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = [s.prompt_template for s in history.strategies]
    if not templates:
        logger.warning("No strategies to compute diversity for.")
        empty = np.zeros((0, 0), dtype=np.float32)
        return empty, output_dir / "strategy_diversity.pdf"

    logger.info("Loading sentence transformer '%s'…", model_name)
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(templates, normalize_embeddings=True)  # shape: (N, D)

    # Cosine similarity = dot product of unit vectors.
    sim_matrix = np.dot(embeddings, embeddings.T)

    labels = [f"Iter {s.metadata.iteration}" for s in history.strategies]

    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(3, len(labels))))
    im = ax.imshow(sim_matrix, vmin=0, vmax=1, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Strategy Prompt Template Similarity")

    # Annotate cells.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    path = output_dir / "strategy_diversity.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Strategy diversity heatmap saved to %s", path)
    return sim_matrix, path


def failure_mode_report(
    history,  # StrategyHistory
    output_dir: Optional[Path] = None,
) -> tuple[str, Path]:
    """
    Generate a failure-mode report: per-type accuracy trends and per-iteration breakdown.

    Returns (report_text, pdf_path). Also writes a .txt report.
    The PDF shows per-type accuracy vs. iteration for each question type.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all question types and accuracy trends from saved eval files.
    all_types: set[str] = set()
    type_trends_data: dict[str, list[tuple[int, Optional[float], Optional[float]]]] = {}
    reflections_by_iteration = {
        strategy.metadata.iteration: reflection
        for strategy, reflection in zip(history.strategies, history.reflections)
        if reflection is not None
    }

    for strategy in history.strategies:
        iteration = strategy.metadata.iteration
        dev_acc_by_type = _load_eval_accuracy_by_type(output_dir, iteration, "dev")
        train_acc_by_type = _load_eval_accuracy_by_type(output_dir, iteration, "train")

        reflection = reflections_by_iteration.get(iteration)
        if not dev_acc_by_type and reflection is not None:
            dev_acc_by_type = dict(reflection.accuracy_by_type)

        combined_types = set(dev_acc_by_type.keys()) | set(train_acc_by_type.keys())
        for q_type in combined_types:
            all_types.add(q_type)
            type_trends_data.setdefault(q_type, []).append(
                (iteration, dev_acc_by_type.get(q_type), train_acc_by_type.get(q_type))
            )

    # Build text report.
    lines = ["=" * 60, "EvoAgent Failure Mode Report", "=" * 60, ""]

    if not history.strategies:
        lines.append("No strategies in history.")
    else:
        best = history.best_strategy()
        dev_acc_str = f"{best.metadata.dev_accuracy:.3f}" if best and best.metadata.dev_accuracy is not None else "—"
        lines.append(
            f"Total iterations: {len(history.strategies)}\n"
            f"Best strategy: iteration {best.metadata.iteration if best else '—'}, "
            f"dev accuracy {dev_acc_str}\n"
        )

        lines.append("Per-type accuracy (from iter eval train/dev files):")
        for q_type in sorted(all_types):
            trend = type_trends_data.get(q_type, [])
            if trend:
                dev_accs = [a for _, a, _ in trend if a is not None]
                train_accs = [a for _, _, a in trend if a is not None]
                dev_summary = "dev=n/a"
                train_summary = "train=n/a"
                if dev_accs:
                    dev_summary = (
                        f"dev min={min(dev_accs):.3f}, max={max(dev_accs):.3f}, "
                        f"final={dev_accs[-1]:.3f}, trend={'↑' if len(dev_accs) > 1 and dev_accs[-1] > dev_accs[0] else '↓' if len(dev_accs) > 1 and dev_accs[-1] < dev_accs[0] else '→'}"
                    )
                if train_accs:
                    train_summary = (
                        f"train min={min(train_accs):.3f}, max={max(train_accs):.3f}, "
                        f"final={train_accs[-1]:.3f}, trend={'↑' if len(train_accs) > 1 and train_accs[-1] > train_accs[0] else '↓' if len(train_accs) > 1 and train_accs[-1] < train_accs[0] else '→'}"
                    )
                lines.append(f"  {q_type}: {dev_summary}; {train_summary}")

        lines.append("\nHypotheses by iteration:")
        for strategy in history.strategies:
            reflection = reflections_by_iteration.get(strategy.metadata.iteration)
            if reflection:
                lines.append(
                    f"  Iter {strategy.metadata.iteration}: {reflection.hypothesis[:200]}"
                )
            else:
                # Last iteration may not have a reflection (skipped to save API calls)
                lines.append(
                    f"  Iter {strategy.metadata.iteration}: (No reflection generated - last iteration optimization)"
                )

    report_text = "\n".join(lines)

    txt_path = output_dir / "failure_mode_report.txt"
    txt_path.write_text(report_text, encoding="utf-8")

    # Build per-type accuracy trend plot with enhanced styling and independent scaling.
    pdf_path = output_dir / "failure_mode_report.pdf"
    if type_trends_data:
        n_types = len(type_trends_data)
        fig, axes = plt.subplots(
            nrows=(n_types + 1) // 2,
            ncols=2,
            figsize=(14, 4 * ((n_types + 1) // 2)),
            squeeze=False,
        )
        axes_flat = [ax for row in axes for ax in row]

        seed_accuracy_by_type = _load_eval_accuracy_by_type(output_dir, 0, "dev")
        if not seed_accuracy_by_type and 0 in reflections_by_iteration:
            seed_accuracy_by_type = dict(reflections_by_iteration[0].accuracy_by_type)

        for ax, (q_type, trend_data) in zip(axes_flat, sorted(type_trends_data.items())):
            trend_data = sorted(trend_data)
            dev_points = [(it, acc) for it, acc, _ in trend_data if acc is not None]
            train_points = [(it, acc) for it, _, acc in trend_data if acc is not None]

            plotted_values: list[float] = []
            if dev_points:
                dev_iters, dev_accs = zip(*dev_points)
                dev_accs_pct = 100 * np.array(dev_accs)
                plotted_values.extend(dev_accs_pct.tolist())
                ax.plot(dev_iters, dev_accs_pct, marker="o", color="royalblue", lw=3, label="Dev", mew=3.5, ms=3.5)
            if train_points:
                train_iters, train_accs = zip(*train_points)
                train_accs_pct = 100 * np.array(train_accs)
                plotted_values.extend(train_accs_pct.tolist())
                ax.plot(train_iters, train_accs_pct, marker="^", color="#F15757", lw=3, label="Train", mew=3.5, ms=3.5)

            seed_baseline = seed_accuracy_by_type.get(q_type)
            if seed_baseline is not None:
                seed_baseline_pct = seed_baseline * 100
                plotted_values.append(seed_baseline_pct)
                ax.axhline(seed_baseline_pct, linestyle="-.", color="#F15757", lw=2, label=f"Seed ({seed_baseline_pct:.3f})", zorder=0)

            title = q_type.replace("_", " ").capitalize()
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xlabel("Iteration")

            local_min = min(plotted_values)
            local_max = max(plotted_values)
            padding = max((local_max - local_min) * 0.1, 2.0)
            ax.set_ylim(local_min - padding, local_max + padding)
            ax.set_xticks(np.array(sorted({it for it, _, _ in trend_data})))

            ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
            ax.legend(prop={"size": 8})

        # Hide empty subplots.
        for ax in axes_flat[n_types:]:
            ax.set_visible(False)

        fig.suptitle("Per-Type Accuracy", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Failure mode report (PDF) saved to %s", pdf_path)
    else:
        logger.info("No reflection data available for per-type trend plot.")

    logger.info("Failure mode report (text) saved to %s", txt_path)
    return report_text, pdf_path
