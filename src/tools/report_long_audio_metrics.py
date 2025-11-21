import argparse
import glob
import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch


def compute_metrics(concat_path: str) -> Dict:
    payload = torch.load(concat_path)
    recon = payload["x_param"].float()
    target = payload["inputs"].float()
    diff = recon - target

    mse = diff.pow(2).mean().item()
    mae = diff.abs().mean().item()
    max_val = target.abs().max().item()
    psnr = 10 * math.log10((max_val**2) / (mse + 1e-12)) if mse > 0 else float("inf")

    duration_ms = sum(payload.get("segment_durations_ms", []))
    chunk_metrics = payload.get("chunk_metrics", [])
    chunk_maes = [cm["mae"] for cm in chunk_metrics] if chunk_metrics else []
    chunk_max_mae = max(chunk_maes) if chunk_maes else None
    chunk_min_mae = min(chunk_maes) if chunk_maes else None

    return {
        "utterance_file": os.path.basename(concat_path),
        "duration_sec": duration_ms / 1000,
        "num_segments": len(payload.get("segment_text", [])),
        "optimization_time_sec": payload.get("time", 0.0),
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "chunk_max_mae": chunk_max_mae,
        "chunk_min_mae": chunk_min_mae,
        "transcript_preview": payload.get("transcript", "")[:120],
        "run_dir": os.path.basename(os.path.dirname(concat_path)),
    }


def gather_metrics(runs_dir: str) -> List[Dict]:
    concat_paths = glob.glob(
        os.path.join(runs_dir, "*", "sampleidx_*_concat.pt"), recursive=True
    )
    metrics = [compute_metrics(path) for path in sorted(concat_paths)]
    return metrics


def save_csv(metrics: List[Dict], out_path: str):
    if not metrics:
        return
    header = list(metrics[0].keys())
    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in metrics:
            values = [str(row[h]) for h in header]
            f.write(",".join(values) + "\n")


def plot_summary(metrics: List[Dict], out_path: str):
    if not metrics:
        return
    indices = list(range(len(metrics)))
    psnr_vals = [row["psnr"] for row in metrics]
    times = [row["optimization_time_sec"] / 60 for row in metrics]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(indices, psnr_vals, "o-", color="tab:blue", label="PSNR (dB)")
    ax1.set_xlabel("Utterance Index")
    ax1.set_ylabel("PSNR (dB)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.bar(indices, times, alpha=0.3, color="tab:orange", label="Time (min)")
    ax2.set_ylabel("Runtime (minutes)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize metrics for long-audio reconstructions."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "long_audio_runs", "10s_plus"
        ),
        help="Directory that contains experiment folders.",
    )
    parser.add_argument(
        "--csv-name", type=str, default="metrics_summary.csv", help="Name of CSV output."
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="metrics_summary.png",
        help="Name of plot output.",
    )
    args = parser.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)
    metrics = gather_metrics(runs_dir)
    if not metrics:
        print(f"No reconstructions found under {runs_dir}")
        return

    csv_path = os.path.join(runs_dir, args.csv_name)
    save_csv(metrics, csv_path)
    plot_summary(metrics, os.path.join(runs_dir, args.plot_name))

    print(f"Wrote {len(metrics)} entries to {csv_path}")
    print(f"Saved plot to {os.path.join(runs_dir, args.plot_name)}")


if __name__ == "__main__":
    main()
