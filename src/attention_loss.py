"""Mixture-of-Layers combination and L_att computation (ISAC.md Sec 2.2, 2.3).

Operates on a *single example* at a time (shape `(num_layers, N1, N2)`
per-layer attention, i.e. `A_l^L`/`A_l^S` stacked across layers), matching
the per-example granularity of `cache_teacher_attention.py` (Sec 3) and
`critical_tokens.py` (Sec 3.2/3.3), since `N1`/`N2` vary per example.
Batch averaging (`E_q`, Sec 2.3) is the caller's (`CODI.forward()`,
Sec 3.5/roadmap step 6) responsibility.
"""
import torch
import torch.nn.functional as F


def aggregate_layer_attention(attn: torch.Tensor, step_groups, critical_groups) -> torch.Tensor:
    """ISAC.md Sec 2.1, Eq 2/3: `A_l[alpha, beta] = sum_{i in K_alpha, j in P_beta} attn[i, j]`.

    `attn` is a single layer's head-averaged attention, shape `(seq, seq)`.
    Shared by `cache_teacher_attention.py` (teacher side) and `CODI.forward()`
    (student side, ISAC.md Sec 3.5) so both sides aggregate identically.
    """
    seq_len = attn.size(0)
    step_indicator = torch.zeros(len(step_groups), seq_len, dtype=attn.dtype, device=attn.device)
    for alpha, indices in enumerate(step_groups):
        if indices:
            step_indicator[alpha, indices] = 1.0

    critical_indicator = torch.zeros(seq_len, len(critical_groups), dtype=attn.dtype, device=attn.device)
    for beta, indices in enumerate(critical_groups):
        if indices:
            critical_indicator[indices, beta] = 1.0

    return step_indicator @ attn @ critical_indicator


def compute_teacher_mol_weights(attn_per_layer: torch.Tensor, tau1: float = 0.1) -> torch.Tensor:
    """Parameter-free, gradient-statistic-based teacher layer weighting
    (ISAC.md Sec 2.2 Eq 4).

    Args:
        attn_per_layer: raw per-layer `A_l^L`, shape `(num_layers, N1, N2)`.
        tau1: teacher MoL softmax temperature (Sec 2.4 default: 0.1).

    Returns:
        `p^L`, shape `(num_layers,)`.
    """
    num_layers, n1, n2 = attn_per_layer.shape
    if n2 < 2:
        # MoLSAKI's G(A_l) is a finite difference across adjacent critical-token
        # columns and is undefined with fewer than 2 critical tokens. Neither
        # paper addresses this edge case (a single-critical-token example);
        # falling back to a uniform layer weighting is the conservative choice
        # here rather than crashing training on it.
        return torch.full(
            (num_layers,), 1.0 / num_layers, dtype=attn_per_layer.dtype, device=attn_per_layer.device
        )

    diffs = (attn_per_layer[:, :, 1:] - attn_per_layer[:, :, :-1]).abs()  # (num_layers, N1, N2-1)
    gradient_stat = diffs.sum(dim=(1, 2)) / (n1 * (n2 - 1))  # (num_layers,)
    return torch.softmax(gradient_stat / tau1, dim=-1)


def combine_layer_attention(attn_per_layer: torch.Tensor, layer_weights: torch.Tensor) -> torch.Tensor:
    """`A = Sum_l p_l * A_l` (ISAC.md Sec 2.2), shared by both the teacher
    (weights from `compute_teacher_mol_weights`) and the student (weights
    from `StudentMoL`, `src/model.py`).

    Args:
        attn_per_layer: shape `(num_layers, N1, N2)`.
        layer_weights: shape `(num_layers,)`.

    Returns:
        Combined attention, shape `(N1, N2)`.
    """
    return torch.einsum("l,lij->ij", layer_weights, attn_per_layer)


def compute_attention_loss(teacher_attn: torch.Tensor, student_attn: torch.Tensor) -> torch.Tensor:
    """`L_att` for a single example (ISAC.md Sec 2.3): row-wise softmax
    normalization of the (already MoL-combined) teacher/student attention,
    then KL divergence averaged over the step axis (`N1`).

    Args:
        teacher_attn: combined `A^L`, shape `(N1, N2)`.
        student_attn: combined `A^S`, shape `(N1, N2)`.

    Returns:
        Scalar loss. The caller averages this over the batch (`E_q`, Sec 2.3).
    """
    if teacher_attn.shape != student_attn.shape:
        raise ValueError(
            f"teacher/student attention shape mismatch: {tuple(teacher_attn.shape)} vs "
            f"{tuple(student_attn.shape)} -- ISAC.md Sec 3.3 guarantees these should match "
            "by construction (same step count and critical-token count)."
        )

    log_teacher = F.log_softmax(teacher_attn, dim=-1)
    log_student = F.log_softmax(student_attn, dim=-1)
    kl_per_step = F.kl_div(log_student, log_teacher, log_target=True, reduction="none").sum(dim=-1)  # (N1,)
    return kl_per_step.mean()
