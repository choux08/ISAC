"""Regression checks for src/attention_loss.py (ISAC.md Sec 2.1-2.3).

Self-contained: uses synthetic per-layer attention tensors, no model/network
download needed. Run directly: `python tests/test_attention_loss.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.attention_loss import (
    aggregate_layer_attention,
    combine_layer_attention,
    compute_attention_loss,
    compute_teacher_mol_weights,
)


def test_aggregate_layer_attention():
    # 4-token sequence: tokens [0,1] are step 0, [2,3] are step 1; token 1 and 3 are critical.
    attn = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )
    step_groups = [[0, 1], [2, 3]]
    critical_groups = [[1], [3]]
    A = aggregate_layer_attention(attn, step_groups, critical_groups)
    assert A.shape == (2, 2)
    # A[0,0] = attn[0,1] + attn[1,1] = 0.2 + 0.3
    assert torch.isclose(A[0, 0], torch.tensor(0.5))
    # A[1,1] = attn[2,3] + attn[3,3] = 0.25 + 0.7
    assert torch.isclose(A[1, 1], torch.tensor(0.95))
    print("OK: aggregate_layer_attention produces the expected step x critical-token sums")


def test_teacher_mol_weights_sum_to_one_and_favor_variable_layers():
    torch.manual_seed(0)
    num_layers, n1, n2 = 4, 3, 5
    attn = torch.rand(num_layers, n1, n2) * 0.01  # low-variation layers
    attn[2] = torch.linspace(0, 1, n2).unsqueeze(0).expand(n1, n2)  # one highly-variable layer
    weights = compute_teacher_mol_weights(attn, tau1=0.1)
    assert weights.shape == (num_layers,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert weights.argmax().item() == 2, "the highly-variable layer should get the most weight"
    print("OK: compute_teacher_mol_weights sums to 1 and concentrates on the high-gradient layer")


def test_teacher_mol_weights_n2_less_than_2_falls_back_to_uniform():
    attn = torch.randn(4, 3, 1)
    weights = compute_teacher_mol_weights(attn, tau1=0.1)
    assert torch.allclose(weights, torch.full((4,), 0.25))
    print("OK: N2<2 edge case falls back to uniform layer weights (ISAC.md Sec 2.2 design decision)")


def test_combine_layer_attention():
    attn = torch.stack([torch.full((2, 2), float(l)) for l in range(3)], dim=0)  # layer l is all-l
    weights = torch.tensor([0.2, 0.3, 0.5])
    combined = combine_layer_attention(attn, weights)
    expected = 0 * 0.2 + 1 * 0.3 + 2 * 0.5
    assert torch.allclose(combined, torch.full((2, 2), expected))
    print("OK: combine_layer_attention computes the weighted sum across layers")


def test_compute_attention_loss_finite_and_gradient_flows():
    teacher = torch.randn(3, 4)
    student = torch.randn(3, 4, requires_grad=True)
    loss = compute_attention_loss(teacher, student)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    print("OK: compute_attention_loss is finite and differentiable w.r.t. the student side")


def test_compute_attention_loss_identical_inputs_gives_zero():
    same = torch.randn(3, 4)
    loss = compute_attention_loss(same, same.clone())
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)
    print("OK: KL(P||P) == 0 (sanity check on the loss formula itself)")


def test_compute_attention_loss_shape_mismatch_raises():
    try:
        compute_attention_loss(torch.randn(3, 5), torch.randn(3, 4))
        raise AssertionError("expected ValueError on shape mismatch")
    except ValueError:
        print("OK: mismatched teacher/student shapes raise a clear ValueError (ISAC.md Sec 3.3)")


if __name__ == "__main__":
    test_aggregate_layer_attention()
    test_teacher_mol_weights_sum_to_one_and_favor_variable_layers()
    test_teacher_mol_weights_n2_less_than_2_falls_back_to_uniform()
    test_combine_layer_attention()
    test_compute_attention_loss_finite_and_gradient_flows()
    test_compute_attention_loss_identical_inputs_gives_zero()
    test_compute_attention_loss_shape_mismatch_raises()
    print("ALL CHECKS PASSED")
