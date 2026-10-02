"""Regression checks for CODI's att_cache_dir/config consistency check
(ISAC.md Sec 3.6.1). Self-contained: builds its own fake att_cache_dir with
just a meta.json (the consistency check only reads that file at __init__
time; it doesn't need real cached attention tensors).

Run directly: `python tests/test_att_cache_consistency.py`.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peft import LoraConfig, TaskType

from src.model import CODI, ModelArguments, TrainingArguments

model_args = ModelArguments(model_name_or_path="gpt2", lora_init=True, train=True)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["c_attn", "c_proj", "c_fc"], init_lora_weights=True,
)


def make_fake_cache_dir(meta: dict) -> str:
    cache_dir = tempfile.mkdtemp()
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return cache_dir


def build(att_cache_dir, critical_token_mode="numeric", include_last_cot=False, teacher_model_name_or_path=None):
    training_args = TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        use_student_mol=True,
        use_att_loss=True,
        att_cache_dir=att_cache_dir,
        critical_token_mode=critical_token_mode,
        include_last_cot=include_last_cot,
        teacher_model_name_or_path=teacher_model_name_or_path,
        print_loss=False,
    )
    return CODI(model_args, training_args, lora_config)


def test_matching_config_succeeds():
    cache_dir = make_fake_cache_dir(
        {"critical_token_mode": "numeric", "include_last_cot": False, "teacher_model_name_or_path": "gpt2"}
    )
    build(cache_dir)  # should not raise
    print("OK: matching critical_token_mode/include_last_cot builds without error")


def test_critical_token_mode_mismatch_raises():
    cache_dir = make_fake_cache_dir(
        {"critical_token_mode": "keyword", "include_last_cot": False, "teacher_model_name_or_path": "gpt2"}
    )
    try:
        build(cache_dir, critical_token_mode="numeric")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "critical_token_mode" in str(e)
        print("OK: critical_token_mode mismatch raises ValueError")


def test_include_last_cot_mismatch_raises():
    cache_dir = make_fake_cache_dir(
        {"critical_token_mode": "numeric", "include_last_cot": True, "teacher_model_name_or_path": "gpt2"}
    )
    try:
        build(cache_dir, include_last_cot=False)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "include_last_cot" in str(e)
        print("OK: include_last_cot mismatch raises ValueError")


def test_missing_meta_json_raises():
    empty_dir = tempfile.mkdtemp()
    try:
        build(empty_dir)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "meta.json" in str(e)
        print("OK: missing meta.json raises ValueError")


def test_teacher_mismatch_warns_but_does_not_raise(capsys=None):
    cache_dir = make_fake_cache_dir(
        {"critical_token_mode": "numeric", "include_last_cot": False, "teacher_model_name_or_path": "gpt2"}
    )
    build(cache_dir, teacher_model_name_or_path="a-different-teacher")  # should not raise
    print("OK: teacher_model_name_or_path mismatch does not raise (metadata-only, warns instead)")


if __name__ == "__main__":
    test_matching_config_succeeds()
    test_critical_token_mode_mismatch_raises()
    test_include_last_cot_mismatch_raises()
    test_missing_meta_json_raises()
    test_teacher_mismatch_warns_but_does_not_raise()
    print("ALL CHECKS PASSED")
