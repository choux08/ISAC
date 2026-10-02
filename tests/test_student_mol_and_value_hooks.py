"""Regression checks for StudentMoL + value-vector hooks (ISAC.md Sec 2.2, 3.4).

Two parts:
  1. GPT-2 (downloads the small gpt2 checkpoint on first run, then cached by HF).
  2. A tiny randomly-initialized GQA Llama config (no checkpoint download) --
     exercises the Llama/Mistral/Qwen branch and the GQA-aware value_dim fix,
     matching this repo's other real target model, Llama-3.2-1B-Instruct.

Run directly: `python tests/test_student_mol_and_value_hooks.py`.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from src.model import CODI, ModelArguments, TrainingArguments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_gpt2_student_mol_registration_and_regression():
    model_args = ModelArguments(model_name_or_path="gpt2", lora_init=True, train=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"], init_lora_weights=True,
    )

    # Regression: flag off -> no student_mol attribute at all, matches plain CODI.
    off_args = TrainingArguments(output_dir=tempfile.mkdtemp(), use_student_mol=False, print_loss=False)
    model_off = CODI(model_args, off_args, lora_config)
    assert not hasattr(model_off, "student_mol")

    on_args = TrainingArguments(output_dir=tempfile.mkdtemp(), use_student_mol=True, mol_tau_student=0.5, print_loss=False)
    model_on = CODI(model_args, on_args, lora_config)
    assert model_on.student_mol.num_layers == 12, "gpt2 has 12 layers"
    assert model_on.get_value_dim(model_on.codi.config) == model_on.dim, "gpt2 (MHA, no GQA) value_dim == hidden_size"

    batch, seq_len, hidden_dim = 2, 7, model_on.dim
    fake_values = torch.randn(model_on.student_mol.num_layers, batch, seq_len, hidden_dim)
    p_s = model_on.student_mol(fake_values)
    assert p_s.shape == (batch, model_on.student_mol.num_layers)
    assert torch.allclose(p_s.sum(dim=-1), torch.ones(batch), atol=1e-5)
    print("OK: gpt2 StudentMoL registration (on/off regression) + forward shape/softmax check")


def test_llama_gqa_value_hooks():
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=8, num_key_value_heads=2,  # GQA: value_dim = 2*(32/8) = 8 != hidden_size
        max_position_embeddings=64,
    )
    tmp_dir = tempfile.mkdtemp()
    AutoModelForCausalLM.from_config(config).save_pretrained(tmp_dir)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(tmp_dir)  # any tokenizer; unused for this check

    model_args = ModelArguments(model_name_or_path=tmp_dir, lora_init=True, train=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=4, lora_alpha=8, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], init_lora_weights=True,
    )
    training_args = TrainingArguments(
        output_dir=tempfile.mkdtemp(), use_student_mol=True, use_att_loss=False, mol_tau_student=0.5, print_loss=False
    )
    codi = CODI(model_args, training_args, lora_config).to(DEVICE)

    expected_value_dim = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
    assert expected_value_dim != config.hidden_size, "test config should actually exercise GQA"
    assert codi.get_value_dim(codi.codi.config) == expected_value_dim
    assert codi.student_mol.rmsnorm.normalized_shape[0] == expected_value_dim

    codi._captured_values = {}
    codi._register_value_hooks()
    input_ids = torch.randint(0, config.vocab_size, (1, 6), device=DEVICE)
    with torch.no_grad():
        codi.codi(input_ids=input_ids, output_attentions=True)

    assert sorted(codi._captured_values.keys()) == [0, 1, 2]
    for v in codi._captured_values.values():
        assert v.shape == (1, 6, expected_value_dim)
    print("OK: Llama/GQA branch -- value_dim correctly resolved from num_key_value_heads, v_proj hooks fire with the right shape")


if __name__ == "__main__":
    test_gpt2_student_mol_registration_and_regression()
    test_llama_gqa_value_hooks()
    print("ALL CHECKS PASSED")
