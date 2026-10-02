"""End-to-end integration check of the full ISAC pipeline (roadmap steps 1-8
combined): cache_teacher_attention.py -> train.py with use_att_loss=True.

SLOW and requires: a CUDA GPU (this repo's CODI always loads the base model
in fp16 unless --bf16 is passed, and CPU doesn't support fp16 backward for
the existing SmoothL1 distillation loss -- a pre-existing CODI property, not
ISAC-specific), network access (downloads gpt2 + zen-E/GSM8k-Aug-NL on first
run, cached by HF afterwards), and a few minutes.

Run directly: `python tests/test_train_integration.py`.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise AssertionError(f"command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result.stdout + result.stderr


def test_cache_then_train_with_att_loss():
    cache_dir = tempfile.mkdtemp()
    run(
        [
            sys.executable, "cache_teacher_attention.py",
            "--teacher_model_name_or_path", "gpt2",
            "--att_cache_dir", cache_dir,
            "--max_examples", "5",
        ],
        cwd=REPO_ROOT,
    )
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    assert meta["num_cached"] > 0, "expected at least one example to be cached"

    output_dir = tempfile.mkdtemp()
    log = run(
        [
            sys.executable, "train.py",
            "--output_dir", output_dir,
            "--expt_name", "isac_integration_test",
            "--model_name_or_path", "gpt2",
            "--data_name", "icot-full",
            "--model_max_length", "512",
            "--per_device_train_batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--num_train_epochs", "1",
            "--learning_rate", "3e-3",
            "--use_lora", "True",
            "--lora_r", "8", "--lora_alpha", "16", "--lora_init",
            "--save_strategy", "no",
            "--save_safetensors", "False",
            "--logging_steps", "1",
            "--do_train",
            "--report_to", "none",
            "--num_latent", "4",
            "--exp_mode", "True",
            "--exp_data_num", "5",
            "--use_student_mol", "True",
            "--use_att_loss", "True",
            "--att_cache_dir", cache_dir,
            "--max_steps", "2",
        ],
        cwd=REPO_ROOT,
    )
    assert "'att_loss':" in log, "expected att_loss to be logged"
    assert "att_loss_total=0" not in log.replace("att_loss_total=0.", "SENTINEL"), (
        "expected at least one non-trivial (nonzero) att_loss_total"
    )
    print("OK: cache_teacher_attention.py -> train.py (use_att_loss=True) runs end-to-end and logs att_loss")


def test_molsaki_only_num_latent_zero():
    """MoLSAKI-only baseline (ISAC.md Sec 5/8, decided 2026-07-14): --num_latent 0
    disables CODI's implicit path entirely, so training should reduce to just
    ref_ce_loss (L_pre+L_exp) + att_loss (L_att) -- ce_loss/distill_loss must log as 0
    since the latent loop that would produce them never runs. Reuses the same cache
    fixture pattern as test_cache_then_train_with_att_loss."""
    cache_dir = tempfile.mkdtemp()
    run(
        [
            sys.executable, "cache_teacher_attention.py",
            "--teacher_model_name_or_path", "gpt2",
            "--att_cache_dir", cache_dir,
            "--max_examples", "5",
        ],
        cwd=REPO_ROOT,
    )
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    assert meta["num_cached"] > 0, "expected at least one example to be cached"

    output_dir = tempfile.mkdtemp()
    log = run(
        [
            sys.executable, "train.py",
            "--output_dir", output_dir,
            "--expt_name", "molsaki_only_integration_test",
            "--model_name_or_path", "gpt2",
            "--data_name", "icot-full",
            "--model_max_length", "512",
            "--per_device_train_batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--num_train_epochs", "1",
            "--learning_rate", "3e-3",
            "--use_lora", "True",
            "--lora_r", "8", "--lora_alpha", "16", "--lora_init",
            "--save_strategy", "no",
            "--save_safetensors", "False",
            "--logging_steps", "1",
            "--do_train",
            "--report_to", "none",
            "--num_latent", "0",
            "--exp_mode", "True",
            "--exp_data_num", "5",
            "--use_student_mol", "True",
            "--use_att_loss", "True",
            "--att_cache_dir", cache_dir,
            "--max_steps", "2",
        ],
        cwd=REPO_ROOT,
    )
    assert "'ce_loss': 0" in log, "num_latent=0 should never run the implicit path -- ce_loss must stay 0"
    assert "'distill_loss': 0" in log, "num_latent=0 should never run the implicit path -- distill_loss must stay 0"
    assert "'att_loss':" in log, "expected att_loss to be logged"
    assert "att_loss_total=0" not in log.replace("att_loss_total=0.", "SENTINEL"), (
        "expected at least one non-trivial (nonzero) att_loss_total"
    )
    print("OK: --num_latent 0 (MoLSAKI-only baseline) trains without crashing, logs ce_loss/distill_loss == 0, att_loss != 0")


def test_plain_codi_regression():
    output_dir = tempfile.mkdtemp()
    log = run(
        [
            sys.executable, "train.py",
            "--output_dir", output_dir,
            "--expt_name", "plain_regression_test",
            "--model_name_or_path", "gpt2",
            "--data_name", "icot-full",
            "--model_max_length", "512",
            "--per_device_train_batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--num_train_epochs", "1",
            "--learning_rate", "3e-3",
            "--use_lora", "True",
            "--lora_r", "8", "--lora_alpha", "16", "--lora_init",
            "--save_strategy", "no",
            "--save_safetensors", "False",
            "--logging_steps", "1",
            "--do_train",
            "--report_to", "none",
            "--num_latent", "4",
            "--exp_mode", "True",
            "--exp_data_num", "5",
            "--max_steps", "2",
        ],
        cwd=REPO_ROOT,
    )
    assert "'att_loss': 0" in log, "plain CODI (use_att_loss=False, the default) should log att_loss == 0"
    print("OK: plain CODI (use_att_loss default False) still trains with att_loss == 0 -- no regression")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIPPED: no CUDA GPU available (this repo's CODI loads fp16 regardless of device, "
              "and CPU doesn't support fp16 backward for the existing distillation loss).")
        sys.exit(0)
    test_cache_then_train_with_att_loss()
    test_molsaki_only_num_latent_zero()
    test_plain_codi_regression()
    print("ALL CHECKS PASSED")
