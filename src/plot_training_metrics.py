"""Plot per-metric training curves from a Hugging Face trainer_state.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_TRAINER_STATE = Path("outputs/sft/checkpoint-310/trainer_state.json")
DEFAULT_OUTPUT_DIR = Path("outputs/eval/sft_metrics")


def _safe_metric_name(metric_name: str) -> str:
    """Convert metric names into filesystem-safe file name fragments."""
    return (
        metric_name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def load_trainer_history(trainer_state_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/eval metric history from trainer_state.json."""
    path = Path(trainer_state_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("log_history", [])

    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    for entry in history:
        step = entry.get("step")
        epoch = entry.get("epoch")

        train_metric_names = {
            key
            for key in entry.keys()
            if key not in {"step", "epoch"} and not key.startswith("eval_")
        }
        if train_metric_names:
            row = {"step": step, "epoch": epoch}
            row.update({key: entry[key] for key in train_metric_names})
            train_rows.append(row)

        eval_metric_names = {key for key in entry.keys() if key.startswith("eval_")}
        if eval_metric_names:
            row = {"step": step, "epoch": epoch}
            row.update({key: entry[key] for key in eval_metric_names})
            eval_rows.append(row)

    return pd.DataFrame(train_rows), pd.DataFrame(eval_rows)


def _plot_metric(
    df: pd.DataFrame,
    metric_name: str,
    title: str,
    output_path: Path,
) -> None:
    """Render a single metric curve to disk."""
    clean_df = df[["step", "epoch", metric_name]].dropna()
    if clean_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(clean_df["step"], clean_df[metric_name], marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Global step")
    ax.set_ylabel(metric_name)
    ax.grid(alpha=0.3)

    latest = clean_df.iloc[-1]
    ax.scatter([latest["step"]], [latest[metric_name]], s=60)
    ax.annotate(
        f"step={int(latest['step'])}\nvalue={latest[metric_name]:.4f}",
        (latest["step"], latest[metric_name]),
        textcoords="offset points",
        xytext=(10, 10),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_trainer_state_metrics(
    trainer_state_path: str | Path,
    output_dir: str | Path,
    run_name: str = "sft",
) -> list[Path]:
    """Plot all numeric train/eval metrics into separate png files."""
    train_df, eval_df = load_trainer_history(trainer_state_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []

    for metric_name in train_df.columns:
        if metric_name in {"step", "epoch"}:
            continue
        safe_metric_name = _safe_metric_name(metric_name)
        output_path = output_root / f"{run_name}_train_{safe_metric_name}.png"
        _plot_metric(train_df, metric_name, f"{run_name.upper()} train: {metric_name}", output_path)
        if output_path.exists():
            generated_paths.append(output_path)

    for metric_name in eval_df.columns:
        if metric_name in {"step", "epoch"}:
            continue
        safe_metric_name = _safe_metric_name(metric_name)
        output_path = output_root / f"{run_name}_{safe_metric_name}.png"
        _plot_metric(eval_df, metric_name, f"{run_name.upper()} eval: {metric_name}", output_path)
        if output_path.exists():
            generated_paths.append(output_path)

    if not train_df.empty:
        train_df.to_csv(output_root / f"{run_name}_train_metrics.csv", index=False)
    if not eval_df.empty:
        eval_df.to_csv(output_root / f"{run_name}_eval_metrics.csv", index=False)

    return generated_paths


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Plot all training metrics from a Hugging Face trainer_state.json file."
    )
    parser.add_argument("--trainer-state", default=str(DEFAULT_TRAINER_STATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="sft")
    args = parser.parse_args()

    generated_paths = plot_trainer_state_metrics(
        trainer_state_path=args.trainer_state,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    print(f"Generated {len(generated_paths)} plots.")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
