"""Standalone: plot a loss_history.csv (written by src/debug_utils.LossHistoryLogger)
into a PNG, without needing to re-run training.

Usage:
    python plot_loss_history.py ~/codi_ckpt/codi_nl_gpt2_isac/gsm8k_gpt2_isac/gpt2/ep_40/lr_0.003/seed_11/loss_history.csv
    # writes loss_plot.png next to the CSV by default; override with -o
    python plot_loss_history.py <csv_path> -o /tmp/my_plot.png
"""
import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path (default: loss_plot.png next to the CSV).")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"{args.csv_path} has no data rows.")

    steps = [int(r["step"]) for r in rows]
    keys = [k for k in rows[0].keys() if k != "step"]

    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 2.2 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        vals = [float(r[key]) for r in rows]
        ax.plot(steps, vals, linewidth=1)
        ax.set_ylabel(key, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"ISAC training losses ({os.path.basename(args.csv_path)})")
    fig.tight_layout()

    output_path = args.output or os.path.join(os.path.dirname(args.csv_path), "loss_plot.png")
    fig.savefig(output_path, dpi=110)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
