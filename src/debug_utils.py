"""Debug-only training instrumentation (not part of ISAC.md's spec).

Kept separate from src/model.py and train.py so the core training logic stays
uncluttered. All the actual debug bookkeeping -- ASCII sparkline rendering,
predicted-vs-target token decoding, and loss-history/CSV logging -- lives here;
model.py and train.py just call into it from a handful of sites.

Nothing in this file affects the loss values, gradients, or training behavior --
it only reads already-computed tensors and prints/writes to disk.
"""
import os

import torch


def ascii_sparkline(values, width=40):
    """Render a list of scalars as a one-line unicode sparkline (e.g.
    '▁▂▂▃▅▇█▆▄▃'), so loss trends are visible directly in the training terminal
    without needing tensorboard / a browser / an SSH tunnel."""
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    sampled = values[-width:]
    lo, hi = min(sampled), max(sampled)
    span = hi - lo
    if span < 1e-12:
        return blocks[0] * len(sampled)
    return "".join(blocks[int((v - lo) / span * (len(blocks) - 1))] for v in sampled)


def decode_preview(tokenizer, logits, target_ids_2d, tag, step, print_enabled, every=50):
    """Decode the first example's predicted-vs-target answer tokens and print them
    side by side. Loss numbers alone don't tell you *what* the model is actually
    producing; this makes that visible directly in the training log.

    Args:
        tokenizer: the (slow) CODI tokenizer -- used only for `.decode()`.
        logits: (batch, seq, vocab) predicted logits.
        target_ids_2d: (batch, seq) target ids, IGNORE_INDEX (-100) masked.
        tag: short label distinguishing which pass this is (e.g. "student answer").
        step: current global training step (from CustomTrainer); skipped if None.
        print_enabled: mirrors CODI's --print_loss flag.
        every: only prints every `every` steps, to avoid flooding stdout.
    """
    if not print_enabled or step is None or step % every != 0:
        return
    try:
        with torch.no_grad():
            row_target = target_ids_2d[0]
            mask = row_target != -100
            if mask.sum() == 0:
                return
            pred_ids = logits[0].argmax(dim=-1)[mask]
            true_ids = row_target[mask]
            vocab_size = len(tokenizer)
            # Model can (esp. early in training) predict one of CODI's 3 appended
            # mem-tokens (pad/bot/eot), which aren't in the tokenizer's own vocab
            # and can't be decoded -- drop those positions rather than crashing.
            pred_ids = pred_ids[pred_ids < vocab_size]
            true_ids = true_ids[true_ids < vocab_size]
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            true_text = tokenizer.decode(true_ids, skip_special_tokens=True)
        print(f"[forward] [{tag}] step={step} predicted={pred_text!r}")
        print(f"[forward] [{tag}] step={step} target=   {true_text!r}")
    except Exception as e:
        print(f"[forward] [{tag}] decode preview failed: {e}")


class LossHistoryLogger:
    """Records loss scalars per logging step to an in-memory history + a CSV file
    (`<output_dir>/loss_history.csv`), and periodically prints an ASCII sparkline
    per key. Used by train.py's CustomTrainer so loss trends are visible without
    tensorboard/network access, and so there's a plain-text record to re-plot
    properly later (matplotlib, a spreadsheet, whatever) if wanted.
    """

    def __init__(self, output_dir, keys=("loss", "ce_loss", "distill_loss", "ref_ce_loss", "att_loss"), sparkline_every=5, plot_every=5):
        self.output_dir = output_dir
        self.keys = list(keys)
        self.sparkline_every = sparkline_every
        self.plot_every = plot_every
        self.history = {k: [] for k in self.keys}
        self.steps = []
        self.csv_path = None
        self.plot_path = None
        self._warned_no_matplotlib = False

    def record(self, step, log_dict, logging_steps):
        self.steps.append(step)
        for key in self.keys:
            if key in log_dict:
                self.history[key].append(log_dict[key])

        if self.csv_path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.csv_path = os.path.join(self.output_dir, "loss_history.csv")
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w") as f:
                    f.write("step," + ",".join(log_dict.keys()) + "\n")
        with open(self.csv_path, "a") as f:
            f.write(f"{step}," + ",".join(str(log_dict[k]) for k in log_dict.keys()) + "\n")

        if step > 0 and step % (logging_steps * self.sparkline_every) == 0:
            print(f"\n[loss trend @ step {step}]")
            for key, vals in self.history.items():
                if vals:
                    print(f"  {key:14s} {ascii_sparkline(vals):40s} last={vals[-1]:.4f} min={min(vals):.4f} max={max(vals):.4f}")
            print()

        if step > 0 and step % (logging_steps * self.plot_every) == 0:
            self._save_plot(step)

    def _save_plot(self, step):
        """Save a PNG line plot of every tracked loss vs. step to
        `<output_dir>/loss_plot.png`, overwritten each time it's called. Since this
        runs on a headless server, this is meant to be scp'd/rsync'd down and viewed
        locally -- no browser/tensorboard/tunnel needed on the server side.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless -- no display/X11 needed on the server
            import matplotlib.pyplot as plt
        except ImportError:
            if not self._warned_no_matplotlib:
                print(
                    "[LossHistoryLogger] matplotlib not installed -- skipping PNG plot "
                    "(sparkline + CSV still work). `pip install matplotlib` to enable it."
                )
                self._warned_no_matplotlib = True
            return

        try:
            fig, axes = plt.subplots(len(self.keys), 1, figsize=(8, 2.2 * len(self.keys)), sharex=True)
            if len(self.keys) == 1:
                axes = [axes]
            for ax, key in zip(axes, self.keys):
                vals = self.history[key]
                if vals:
                    ax.plot(self.steps[: len(vals)], vals, linewidth=1)
                ax.set_ylabel(key, fontsize=9)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("step")
            fig.suptitle(f"ISAC training losses (up to step {step})")
            fig.tight_layout()

            if self.plot_path is None:
                os.makedirs(self.output_dir, exist_ok=True)
                self.plot_path = os.path.join(self.output_dir, "loss_plot.png")
            fig.savefig(self.plot_path, dpi=110)
            plt.close(fig)
        except Exception as e:
            print(f"[LossHistoryLogger] failed to save loss_plot.png: {e}")
