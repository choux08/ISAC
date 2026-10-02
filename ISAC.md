# ISAC.md

This document is the design/implementation plan for building `ISAC idea.pdf` (Implicit Stepwise Attention in Continuous Space) on top of this repository (the official CODI implementation). All code in this folder (`src/model.py`, `train.py`, `test.py`, `probe_latent_token.py`, `scripts/*`) is the original CODI paper's implementation, and none of it reflects ISAC yet.

This revision resolves two of the previously open issues — the **critical-token definition** and the **teacher/student tokenizer-alignment problem** — using the original `MoLSAKI Yao Chen.pdf` (EMNLP 2025, Chen et al.) as the source. Remaining risks are listed in §7.

**This document is a specification. The code itself will be implemented by a separate session (Claude Code) based on this document.**

## Scope

This document specifies the target implementation.

The objective is to faithfully implement this specification, not to improve or redesign the algorithm.

If an implementation detail appears ambiguous, ask for clarification instead of making assumptions.

## Project Status

This document contains three different kinds of information.

### Confirmed
Information verified from:
- The CODI paper
- The MoLSAKI paper
- The current CODI source code

### Design Decisions
Implementation choices made specifically for the ISAC project. These are not claimed by the original papers (e.g., restricting the first-phase target dataset to `icot-full` in §3.2, or the recommended default hyperparameters in §2.4 being carried over from MoLSAKI's own experimental settings rather than re-tuned for this codebase).

### Assumptions
Reasonable assumptions made because the papers do not specify an implementation detail (e.g., the accuracy hypothesis in §5/§7.3 — that ISAC can reach MoLSAKI-level accuracy at CODI-level latency — has not been tested anywhere).

Whenever an assumption is replaced by experimental evidence, this document should be updated.

## Design Philosophy

The purpose of ISAC is to extend CODI, not to redesign it.

Implementation should follow these principles:

1. Reuse existing CODI modules whenever possible.
2. Preserve the original latent reasoning pipeline.
3. Introduce only the components required by MoLSAKI.
4. Prefer additive changes over replacing existing implementations.
5. Every new module should have a clear correspondence to the ISAC formulation.

---

## 1. What ISAC Adds to CODI

CODI runs a single model through two paths.

- Explicit path (based on `ref_input_ids`, `src/model.py` L296-298, L385-391): the question + natural-language CoT + answer is forwarded as-is. This produces the teacher CE loss (`ref_ce_loss`).
- Implicit path (`encoder_input_ids` → latent loop → `decoder_input_ids`, L279-382): the question is encoded, then `num_latent` continuous latents are generated autoregressively, and finally the answer is decoded. This produces the CE loss (`ce_loss_total`) plus a hidden-state self-distillation loss (`distill_loss_total`) against the explicit path.

In other words, **what CODI calls the "teacher" is not a separate LLM — it is the same model's explicit-CoT forward pass.** This is self-distillation, not distillation from an actual external teacher.

ISAC adds a **real teacher LLM** (a separate, larger, more capable model) whose attention pattern is distilled into the explicit path via MoLSAKI's `L_att` term. So ISAC's two paths are:

- `f_e` (explicit path) = the same forward pass as CODI's explicit/ref path. **Teacher LLM ↔ f_e attention distillation (`L_att`)** is newly attached here.
- `f_i` (implicit path) = identical to CODI's implicit/latent path. The existing `distill_loss` (hidden-state self-distillation, corresponding to ISAC's `L_KD`) is reused as-is.

Final objective (ISAC idea.pdf, Eq. 17):

```
L_ISAC = c_ce,e * L_ce,e + c_ce,i * L_ce,i + c_att * L_att + c_KD * L_KD
```

- `L_ce,e` = CODI's `ref_ce_loss` (L390)
- `L_ce,i` = CODI's `ce_loss_total` (L381-382)
- `L_KD` = CODI's `distill_loss_total` (L354-373) — **already implemented, reuse as-is**
- `L_att` = **needs new implementation**: KL divergence of stepwise attention between the teacher LLM and `f_e` (the explicit path)

So the core of the ISAC implementation is **adding a single new term, `L_att`**. All other loss terms in the CODI code can be used as they are.

---

## 2. `L_att` (MoLSAKI Distillation) — Formulas and Detailed Definitions per the Original Papers

Consolidated from ISAC idea.pdf §3.2 and the original MoLSAKI paper (Chen et al., 2025) §3.2–3.3.

### 2.1 Stepwise / Critical-Token Attention Matrices

Let `I_l^L ∈ R^{M×M}` (teacher) and `I_l^S ∈ R^{N×N}` (student) be the `l`-th layer's head-averaged self-attention matrix for the teacher LLM and student (`f_e`) respectively (M, N are each model's own tokenizer-based sequence length — these may differ; §3.2 explains why this doesn't matter).

- `M1` (teacher), `N1` (student): the collection of token-index sets partitioned by reasoning step. **MoLSAKI splits these on periods (`.`)** ("we segment the input sequence composed of question and rationale into reasoning steps based on periods"). The question itself is also counted as a step.
- `M2` (teacher), `N2` (student): the collection of critical-token index sets. **Extracted via regex matching plus the tokenizer's char-to-token mapping** ("Utilizing regular expression matching and the tokenizer's mapping, we obtain the index set of critical tokens").
- Step × critical-token attention matrix (MoLSAKI Eq. 2, 3 / ISAC idea.pdf Eq. 3, 4):
  ```
  A^L_l[α, β] = Σ_{i∈K_α^L, j∈P_β^L} I_l^L[i, j]      (K^L∈M1, P^L∈M2)
  A^S_l[α, β] = Σ_{i∈K_α^S, j∈P_β^S} I_l^S[i, j]      (K^S∈N1, P^S∈N2)
  ```
  For each critical-token column, sum the rows belonging to each step — i.e., aggregate, per step, "how much attention every token paid to this specific critical token."
  Result: `A^L_l ∈ R^{|M1|×|M2|}`, `A^S_l ∈ R^{|N1|×|N2|}`, and by construction **`|M1|=|N1|` (same number of steps in the same rationale) and `|M2|=|N2|` (same number of critical tokens)** (this is the key point of §3.3).

### 2.2 Mixture-of-Layers (MoL) — Resolving the Layer-Count Mismatch

- Teacher: compute each layer's "step-to-step variation (gradient)" of attention (MoLSAKI Eq. 4):
  ```
  G(A_l^L) = Σ_i Σ_j |A_l^L[i,j+1] - A_l^L[i,j]| / (|M1| × (|M2|-1))
  p^L = softmax([G(A_1^L), ..., G(A_{L1}^L)], τ1)   ∈ R^{L1}
  A^L = Σ_l p_l^L · A_l^L
  ```
  **Gradient-free, statistic-based. No trainable parameters.**
  **Implemented** as `compute_teacher_mol_weights` in `src/attention_loss.py`. **Design decision (undefined by MoLSAKI):** when `|M2| < 2`, `G(A_l)`'s finite difference is undefined; the implementation falls back to a uniform layer weighting (`1/L1` each) rather than raising, so a single-critical-token example doesn't crash training. Revisit if this proves to matter empirically.
- Student: RMSNorm each layer's value matrix `V_l`, sum over the sequence dimension to get a layer embedding `h_l`, concatenate, and pass through a trainable linear + softmax (temperature `τ2`) (MoLSAKI Eq. 5):
  ```
  h_l = Σ_i RMSNorm(V_l)[i,:]
  H = concat(h_1, ..., h_{L2})
  p^S = softmax(H·W + b, τ2)   ∈ R^{L2}
  A^S = Σ_l p_l^S · A_l^S
  ```
  **Requires trainable parameters (`W`, `b`) — to be optimized jointly with the LoRA parameters.**
  **Implemented** as `StudentMoL` in `src/model.py`. **Design decision (undefined by MoLSAKI):** the equation doesn't say whether RMSNorm's learnable scale is shared across layers or independent per layer. Implemented as a **single shared** `nn.RMSNorm`, on the reasoning that the whole point of normalizing before the layer-comparison step is to remove layer-specific scale differences -- a per-layer affine scale would partially undo that. Revisit if empirically the layers turn out to need independent normalization.
  **Implementation note (not covered by MoLSAKI or the original ISAC spec):** `V_l`'s width is **not** always `hidden_size`. For GQA models (`num_key_value_heads < num_attention_heads`, e.g. Llama-3.2-1B-Instruct, Mistral, Qwen2), the value projection's output dim is `num_key_value_heads * head_dim`, which is smaller than `hidden_size`. `StudentMoL` must be sized with this GQA-aware value dim (see `CODI.get_value_dim()`), not `self.dim`.

### 2.3 Final `L_att`

After row-wise softmax normalization (MoLSAKI Eq. 6 / ISAC idea.pdf Eq. 13, 14), the KL divergence averaged over the step axis:
```
Ã^L[i,:] = softmax(A^L[i,:])
Ã^S[i,:] = softmax(A^S[i,:])
L_att = E_q[ (1/N1) Σ_i KL( Ã^L[i,:] ‖ Ã^S[i,:] ) ]
```

### 2.4 Default Hyperparameters Validated by MoLSAKI

Recommended starting values based on MoLSAKI's §4.1/§4.3 experimental results (GPT-2/TinyLlama-scale student, Llama3-8B/Qwen2.5-32B teacher):

| Hyperparameter | Paper notation | Recommended starting value | Notes |
|---|---|---|---|
| `L_att` weight | `β` (→ ISAC's `c_att`) | `1.0` | On SVAMP, performance peaks at β=1.0; performance drops if β increases further (Fig. 5) |
| Teacher MoL temperature | `τ1` | `0.1` | As τ1 increases, the teacher fails to concentrate on specific (middle) layers and performance degrades |
| Student MoL temperature | `τ2` | `0.5` | Too small (over-concentration) or too large (over-uniformity) both hurt performance; ~0.5 is the peak on SVAMP/AsDiv |

Adaptive layer alignment (MoL) consistently outperforms fixed single-layer mapping (SL) in ablations (MoLSAKI Table 2) — **do not hand-pick layers; a trainable MoL must be implemented.**

---

## 3. New Components to Implement

### 3.1 Teacher LLM Loader + Attention Extraction
- Near `CODI.__init__` in `src/model.py` (L138~), a separate teacher model (`AutoModelForCausalLM`, frozen, `output_attentions=True`) needs to be held.
- The teacher **may use a different tokenizer** from the student (`self.codi`) — as explained in §3.2, MoLSAKI's design explicitly targets "no shared tokenizer/vocabulary required." So teacher selection should be driven by **reasoning ability and rationale quality**, not tokenizer compatibility (MoLSAKI §4.6: it's even possible to decouple the rationale-generation teacher from the attention-extraction teacher — e.g., use a GPT-4-class model for rationale, and a separate open-source teacher (e.g., Llama3-8B) for attention extraction).
- The teacher forward pass is not a training target, so use `torch.no_grad()` + `eval()`. Per-step online forward is costly, so **caching teacher attention during a preprocessing stage is adopted as the default approach** (§7.1).

**Implemented (deviates from the paragraph above): CODI never loads a teacher model at all.** Given §7.1 explicitly adopts caching as the default, the implementation goes one step further: `cache_teacher_attention.py` is a fully separate offline script that loads the teacher, forwards `question+rationale` through it, and writes one `{raw_index}.pt` file per example (raw per-layer `A_l^L`, keyed by the example's index in the underlying HF dataset) plus a `meta.json`. `CODI` (`src/model.py`) only ever reads `att_cache_dir` at train time via `torch.load` -- it has no code path that loads a teacher checkpoint. `TrainingArguments.teacher_model_name_or_path` is kept only as a **provenance/metadata field** (checked against `att_cache_dir/meta.json`, see §3.6.1) and is never passed to `AutoModelForCausalLM.from_pretrained`. Rationale: avoids holding two models in GPU memory simultaneously during training (relevant on small-VRAM setups) and matches §7.1's stated cost concern.

### 3.2 Step / Critical-Token Index Definitions (`K_α`, `P_β`) — Finalized per Dataset
Concretized here based on the MoLSAKI original text (resolving the "undefined" status of the previous version):

- **Step segmentation (`M1`/`N1`)**: split on periods (`.`). CODI's `train.py` already splits CoT on `". "` for the `icot-full` (natural-language CoT, `zen-E/GSM8k-Aug-NL`) data — **reuse as-is**. The `icot` (word-split, `zen-E/GSM8k-Aug`) variant splits by word rather than sentence and does not match MoLSAKI's step definition, so **ISAC's first-phase target dataset is recommended to be restricted to `icot-full` (GSM8K-Aug-NL)**.
- **Critical-token selection (`M2`/`N2`)**: splits into two branches depending on dataset type (MoLSAKI §3.2 + Figures 1/2).
  - **Math reasoning (GSM8K-Aug-NL)**: define **numeric tokens** as critical tokens. Find numeric spans in the raw text via regex (e.g., `\d+(\.\d+)?`), and map them to token indices using the tokenizer's char↔token mapping (`offset_mapping`, or a fast tokenizer's `encode_plus(..., return_offsets_mapping=True)`).
  - **Commonsense reasoning (CommonsenseQA/StrategyQA)**: define a **keyword list** as critical tokens (e.g., key nouns/entities pulled from the question/choices, as in MoLSAKI Figure 2a). CODI's `zen-E/CommonsenseQA-GPT4omini` data has no keyword labels, so an additional preprocessing step is needed to ask the teacher LLM to extract keywords via few-shot prompting (the original paper obtains these keywords the same way, via teacher/LLM prompting).
- Teacher and student each apply this rule (period splitting, regex/keyword matching) **independently to their own tokenization** — since both are applied to the same underlying text (question + rationale), `|M1|=|N1|` and `|M2|=|N2|` hold automatically (see §3.3).

### 3.3 The Tokenizer-Alignment Problem Does Not Arise by Design (Resolves the Previous Version's §5.2 Issue)
MoLSAKI explicitly states "eliminating the requirement for shared tokenizers or vocabularies" as a design goal. Reasoning:
- `A^L` and `A^S` are defined **in (step, critical-token) index space, not token-index space**. The teacher's raw attention `I_l^L` is `M×M` (teacher token count) and the student's `I_l^S` is `N×N` (student token count) — these can differ freely. Once each is **aggregated (summed)** down to "step × critical-token" size, both matrices end up with the **same shape**, `|M1|×|M2| = |N1|×|N2|`.
- The only requirement is that "applying the same rule (period splitting / regex-keyword matching) independently to teacher and student for the same underlying text yields the same step count and the same critical-token count." Because the rule is defined at the text (string) level while token-level mapping is handled independently by each tokenizer, this holds automatically.
- Therefore **it is fine for the teacher and student to be from entirely different model families (GPT-2 vs. Llama-3 vs. Qwen)** — teacher choice can be driven purely by performance/cost rather than tokenizer compatibility.
- Practical caveats: (a) a critical token should never be found in only one of the teacher/student texts (e.g., due to differing numeral formatting or how "12" gets tokenized) — to avoid this, apply the regex once to the raw text and then map the result to each tokenizer separately. (b) if the rationale text itself differs between teacher and student (e.g., a rationale generated specifically for the teacher differs from the rationale used to train the student), step counts could diverge — this does not apply to ISAC/CODI, since both paths are fed the same rationale from the same dataset.

### 3.4 Mixture-of-Layers (MoL) Module
- Teacher side: a gradient-free, statistics-based weighting function `compute_teacher_mol_weights(attn_per_layer, tau1=0.1)` (§2.2, MoLSAKI Eq. 4).
- Student side: a small trainable submodule `StudentMoL(nn.Module)` — value-vector RMSNorm → concat → Linear(`W`, `b`) → softmax(`tau2=0.5`) (§2.2, MoLSAKI Eq. 5). Register it in `CODI.__init__` as `self.student_mol = StudentMoL(...)` so it trains alongside the LoRA parameters.
- **Do not hard-map layers — a trainable MoL must be implemented** (§2.4, per the MoLSAKI Table 2 ablation).

### 3.5 Computing `L_att` and Integrating It into `forward()`
- Inside `src/model.py`'s `forward()` (L262-409), right after the explicit/ref pass (L296-298), add the teacher forward (or cache load) + `A^L`, `A^S` computation + KL loss.
- Modify the final `loss = ce_loss_total + distill_loss_total + ref_ce_loss` (L400) to add `+ att_loss_total`.
- Log `att_loss_total` via `.detach().item()`, following the same pattern as `distill_loss_total` (L402-407).

**Implemented (deviates from "right after the explicit/ref pass" above): `L_att` does NOT reuse `ref_outputs_with_grad`'s forward pass.** Instead, `forward()` runs a **separate, dedicated forward pass** on `question+rationale` alone (via a fast tokenizer, `add_special_tokens=False`), mirroring exactly what `cache_teacher_attention.py` does for the teacher side. Reason: `ref_input_ids` is built by `train.py`'s `preprocess()` as `question + EOS + cot + EOS + answer` (token-level concatenation with EOS delimiters) using the *slow* tokenizer, whereas critical-token extraction (`get_step_and_critical_token_indices`) needs a *fast* tokenizer's char-offset mapping applied to the raw `question+rationale` string with no delimiters. Reconstructing the exact token offset of `question+rationale` inside the EOS-decorated `ref_input_ids` sequence was judged too fragile; a clean, separate forward call sidesteps the alignment problem entirely. Both calls go through the same shared `self.codi` LoRA weights, so gradients from `L_att` still update `f_e`'s parameters -- just not via the literal `ref_outputs_with_grad` tensor.
- This requires the data pipeline to thread three extra fields into `forward()`: `raw_index`, `question_texts`, `rationale_texts` (plain Python lists, not tensors -- added to `DataCollatorForSupervisedDataset`'s output in `train.py`; HF `Trainer._prepare_input` passes non-tensor values through untouched).
- The student's per-layer attention (`I_l^S`) and value vectors (`V_l`, for `StudentMoL`) are captured from this one extra forward pass: `output_attentions=True` gives the former; forward hooks registered in `CODI._register_value_hooks()` (on `transformer.h[i].attn.c_attn` for GPT-2, `model.layers[i].self_attn.v_proj` for the Llama/Mistral/Qwen family) capture the latter. See §3.4 for the GQA value-dim caveat.
- **Attention-backend note:** `CODI.__init__` loads `self.codi` with no `attn_implementation`, so it defaults to SDPA (transformers 4.52.4 + torch 2.7), which cannot emit attention weights. The `output_attentions=True` request here relies on 4.52.4's automatic per-forward eager fallback (attentions returned, gradients preserved) — so the main training forwards keep SDPA speed and only this L_att forward pays the eager cost. **Do not** force global `attn_implementation="eager"` on `self.codi` (it would slow every main forward for no benefit). Smoke-test `.attentions is not None` for any new student model before a long run; load the student with eager only if a future transformers version drops the fallback. See CLAUDE.md's "ISAC experiment plan" for the run-note version of this.

### 3.6 New CLI Arguments (add to the `TrainingArguments` dataclass around `src/model.py` L84)
| Argument | Description | Recommended default | Correspondence |
|---|---|---|---|
| `teacher_model_name_or_path` | Path to the teacher LLM used for attention distillation | - | New |
| `att_loss_factor` (`c_att`) | Weight of `L_att` | `1.0` (MoLSAKI's β) | New, following the same pattern as `distill_loss_factor`/`ref_loss_factor` |
| `kd_loss_factor` (`c_KD`) | Reuses the existing `distill_loss_factor` | keep existing value | Reuse existing |
| `mol_tau_teacher` (`τ1`) | Teacher MoL softmax temperature | `0.1` | New |
| `mol_tau_student` (`τ2`) | Student MoL softmax temperature | `0.5` | New |
| `critical_token_mode` | `"numeric"` (math) or `"keyword"` (commonsense) | depends on dataset | New |
| `att_cache_dir` | Storage path for teacher attention cached during preprocessing | - | New |
| `use_att_loss` | Master gate for the whole `L_att` path | `False` | New, not in the original spec table -- added so plain-CODI runs are byte-for-byte unaffected by default (mirrors the existing `use_prj` pattern) |
| `use_student_mol` | Registers `StudentMoL` (required when `use_att_loss=True`) | `False` | New, same reasoning as `use_att_loss` |

#### 3.6.1 `att_cache_dir` / config consistency check (hardening, not in the original spec)

Nothing in §3.1-3.6 specifies what happens if `att_cache_dir` was built with a different `critical_token_mode` or `include_last_cot` than the current training run uses. Since `A^L`'s and `A^S`'s shapes only *coincidentally* need to match (§3.3) rather than being cross-validated by content, a silent mismatch (e.g. cache built with `include_last_cot=False`, training run launched with `include_last_cot=True`) could produce a same-shaped but *semantically misaligned* `A^L`/`A^S` pair that trains without ever raising an error.

**Implemented:** `cache_teacher_attention.py` already writes `critical_token_mode`/`include_last_cot`/`teacher_model_name_or_path` into `att_cache_dir/meta.json` (§3, "Teacher LLM Loader"). `CODI.__init__`, when `use_att_loss=True`, now reads this `meta.json` and hard-fails at construction time (before training starts) if `critical_token_mode` or `include_last_cot` disagree with the current `TrainingArguments`, and warns (non-fatal) if `teacher_model_name_or_path` disagrees. `data_name` is not cross-checked here since `CODI.__init__` doesn't receive `data_args`; `cache_teacher_attention.py` restricting itself to `icot-full` (§3.2) is the only guard against a truly wrong dataset.

---

## 4. Mapping to Existing Code

| ISAC concept | Location in current CODI code | Status |
|---|---|---|
| `f_e` (explicit path) forward | `src/model.py` L296-298, L385-391 | Already exists |
| `f_i` (implicit path) forward | `src/model.py` L330-382 | Already exists |
| `L_ce,e` | `ref_ce_loss`, L390 | Already exists |
| `L_ce,i` | `ce_loss_total`, L381-382 | Already exists |
| `L_KD` (self-distillation) | `distill_loss_total`, L354-373 | Already exists, reuse as-is |
| `L_att` | none | **New implementation** |
| Teacher LLM loading | none (CODI has no external teacher) | **New implementation** |
| CoT step splitting (period-based) | `train.py`'s `icot-full` branch (`". "` split) | **Already exists and matches MoLSAKI's definition — reuse as-is** |
| Critical-token (numeric/keyword) extraction | none | **New implementation** (§3.2) |
| MoL (teacher/student) | none | **New implementation** |

---

## 5. CODI-only vs. MoLSAKI-only vs. ISAC

| Aspect | CODI-only | MoLSAKI-only | ISAC (combined) |
|---|---|---|---|
| Reasoning path used at inference | Continuous latent (implicit) path only, no intermediate tokens generated | Generates the explicit natural-language rationale as-is (explicit CoT) | Both explicit and implicit paths used during training; **only the implicit path is used at inference** (keeps CODI's latency advantage) |
| Teacher required | None (the same model's explicit pass acts as a self-teacher) | A separate, large teacher LLM is required (source of attention extraction) | A separate teacher LLM is required (same as MoLSAKI, used for attention extraction) |
| Core losses | `L_ce,e` (teacher CE) + `L_ce,i` (student CE) + `L_KD` (explicit↔implicit hidden-state self-distillation) | `L_pre` (answer CE) + `L_exp` (rationale CE) + `L_att` (teacher↔student stepwise attention KL) | `L_ce,e` + `L_ce,i` + `L_att` (teacher↔explicit path) + `L_KD` (explicit↔implicit path) — all four terms combined |
| Distillation target | Hidden states (per layer, at the position right before the answer) | Attention patterns (step × critical-token, aligned across layers via MoL) | Attention patterns (teacher→explicit) **plus** hidden states (explicit→implicit) — double distillation |
| Inference latency | Very low (only latent tokens generated, no text generation) | High (must generate the full explicit CoT text) | Very low (only the implicit path is used at inference — same latency as CODI) |
| Accuracy tendency (general observation) | Some accuracy loss vs. explicit CoT (a common limitation of implicit CoT), partially mitigated by self-distillation | Achieves state-of-the-art student accuracy among CoT-distillation methods (especially critical-token utilization), but no latency benefit | Goal is MoLSAKI-level accuracy (internalizing critical-token attention) at CODI-level latency — **still unverified (see §7 risks); this is the core experimental goal of this repository** |
| Training cost | No teacher needed, only self-distillation — relatively cheap | Additional cost from teacher LLM attention extraction (typically preprocessing or online forward) | MoLSAKI's teacher cost plus CODI's self-distillation cost combined — the most expensive of the three |
| Implementation status in this repo | Fully implemented (`src/model.py`) | Not implemented (no teacher loader, no critical-token extraction, no MoL) | Only `L_KD` can be reused from existing CODI; `L_att` and its supporting components all need new implementation |

**Summary**: CODI is a "fast but somewhat less accurate than explicit CoT" implicit method, and MoLSAKI is an "accurate but still requires generating explicit tokens" explicit method. ISAC applies MoLSAKI's attention distillation to CODI's explicit path and transfers that capability into the implicit path via self-distillation (`L_KD`), with the project's hypothesis being that **inference can stay CODI-like (latent-only) while accuracy approaches MoLSAKI's level.**

---

## 6. Implementation Roadmap (Step by Step)

**Steps 1-8 are DONE** (implemented and verified against real GPT-2 + real `zen-E/GSM8k-Aug-NL` data, on GPU, in one working session -- see each step below for the file and any deviation from this spec; deviations are cross-referenced into §2.2/§3.1/§3.4/§3.5/§3.6.1 above). Steps 9-10 are not yet done.

1. **DONE. Finalize the first-phase target dataset**: `icot-full` (GSM8K-Aug-NL) — its step splitting is already period-based, matching MoLSAKI's definition directly (§3.2).
2. **DONE. Implement the critical-token extractor**: `src/critical_tokens.py` (`get_step_and_critical_token_indices`, `split_steps_with_offsets`, `find_numeric_critical_spans`, `char_spans_to_token_indices`). `critical_token_mode="numeric"` only; `"keyword"` raises `NotImplementedError` (commonsense keyword-extraction preprocessing is deferred -- needs a chosen teacher LLM, §8).
3. **DONE. Teacher attention extraction/caching script**: `cache_teacher_attention.py` (repo root, alongside `train.py`/`test.py` -- not under `scripts/`, to match this repo's existing convention of `.py` entry points at root + `.sh` launchers under `scripts/`). Caches raw per-layer `A_l^L` (MoL combination deliberately deferred to train time, see step 5) plus a `meta.json`, keyed by the example's raw index in the underlying HF dataset.
4. **DONE. `StudentMoL` module**: `src/model.py`, registered in `CODI.__init__` behind `use_student_mol` (new flag, not in §3.6's original table -- added so plain-CODI runs are unaffected by default).
5. **DONE. `compute_teacher_mol_weights` / `compute_attention_loss`**: `src/attention_loss.py`, along with `combine_layer_attention` (shared teacher/student layer-combination helper, needed by both but not separately named in §3.4/2.2) and `aggregate_layer_attention` (moved here from `cache_teacher_attention.py` so `CODI.forward()` can reuse the exact same aggregation as the caching script).
6. **DONE. `L_att` integration**: `src/model.py`'s `CODI.forward()`, behind `use_att_loss`. See §3.5 for the "separate forward pass, not `ref_outputs_with_grad`" deviation, and §3.1 for the "cache-only, no live teacher" deviation. All new `TrainingArguments` fields from §3.6 added, using the §2.4 defaults.
7. **DONE. `att_loss` logging + data-pipeline wiring**: `train.py`'s `CustomTrainer.compute_loss` (logging) plus additive changes to `preprocess()`/`SupervisedDataset`/`DataCollatorForSupervisedDataset` to thread `raw_index`/`question_texts`/`rationale_texts` through to `forward()` (needed for step 6's separate forward pass; not explicitly anticipated by this spec's original step 7 wording, which only mentioned logging).
8. **DONE. New training scripts**: `scripts/train_gpt2_gsm8k-aug-nl_isac.sh`, `scripts/train_llama1b_gsm8k-aug-nl_isac.sh` (variants of the existing `icot-full` scripts, since that's ISAC's only supported dataset). `TEACHER_MODEL_NAME_OR_PATH`/`ATT_CACHE_DIR` are left as placeholders (teacher choice is still §8, out of scope).
9. **Not done. Validate**: on a small subset (`exp_mode=True`), confirm training runs, `L_att` decreases, and the teacher/student attention matrix shapes (`|M1|=|N1|`, `|M2|=|N2|`) actually match, as a sanity check. (Note: a *mechanical* version of this -- shapes matching, `L_att` finite, gradients flowing -- was already verified during steps 5-7's implementation; step 9 as originally scoped means a longer run checking `L_att` actually trends downward over many steps, which has not been done.)
10. **Not done. Evaluate**: compare GSM8K (etc.) accuracy against the CODI-only baseline using `test.py`. Where possible, also compare against a MoLSAKI-only baseline (explicit-only; not present in this repo, so either reproduce it separately or cite reference numbers) to test the §5 hypothesis.

---

## 7. Remaining Risks and Open Issues

### 7.1 Training Cost (carried over from the previous §5.3, still unresolved)
If a full teacher LLM forward pass (plus attention storage) is required at every training step, the memory/speed overhead is significant. Alternatives:
- (a) Compute teacher attention once during a preprocessing stage and cache it to disk — **recommended as the default direction**.
- (b) Online teacher forward at every step (simpler to implement, but slow).

**Resolved (direction (a) chosen)**: `cache_teacher_attention.py` implements the offline caching path; `CODI` never holds a teacher model (§3.1). **New, still-open sub-risk**: `L_att`, as implemented, still costs one extra small forward pass through the *student* per batch item per step (needed to get `f_e`'s own attention/value vectors, §3.5) -- with this repo's actual training scripts using `per_device_train_batch_size` up to 64, that's up to 64 extra forwards/step. Not yet measured; likely the next thing to profile before a real training run.

### 7.2 Dataset Requirements (carried over from the previous §5.4, partially resolved)
`zen-E/GSM8k-Aug-NL` has no teacher attention or critical-token labels, so a preprocessing pipeline that runs teacher LLM inference once is still needed. However, since the critical-token definition itself is now finalized in §3.2, "what needs to be labeled" is no longer undetermined.

**Resolved**: `cache_teacher_attention.py` is this preprocessing pipeline.

### 7.3 The Accuracy Hypothesis Is Unverified
The §5 claim — "ISAC = MoLSAKI's accuracy + CODI's latency" — is a **proposal-stage hypothesis** with no supporting experiments in ISAC idea.pdf. MoLSAKI has been validated on an explicit model, and CODI has been validated without attention distillation, but whether self-distillation (`L_KD`) actually propagates attention information all the way into the implicit path when the two techniques are combined can only be confirmed by running the experiment in this repository.

**Still unverified** -- roadmap steps 9-10 (not done this session) are what would test this.

### 7.4 Critical-Token Rules Are Not Yet Defined for `icot` (Word-Split, Non-NL) Data
§3.2 restricts the first-phase scope to `icot-full`. Extending to `zen-E/GSM8k-Aug` (word-split) or the commonsense data will require defining separate step/critical-token rules.

**Still open.**

### 7.5 No Committed Regression Coverage (new)
All of steps 2-8's implementation was verified with ad hoc scripts during the implementing session (real GPT-2 forward passes, real cached attention, a real few-step `train.py` run with `use_att_loss=True`), but those scripts lived only in a scratch directory, not in the repo. `tests/` now holds a committed, repo-permanent version of that same coverage (self-contained where possible; the end-to-end one needs network + GPU and is clearly marked slow).

---

## 8. Out of Scope for This Document

- Actual code changes (this document is a plan; writing the code is separate work).
- Selecting a specific teacher LLM model (per §3.1, no tokenizer constraint). **Decided 2026-07-10:** anchor pair = teacher `Qwen/Qwen2.5-7B-Instruct` → student `meta-llama/Llama-3.2-1B-Instruct`, with `Qwen2.5-7B → gpt2` (2nd) and `Llama-3.1-8B → Llama-3.2-1B` (3rd) as the follow-up pairs. See CLAUDE.md's "ISAC experiment plan" for the full ranking and rationale (the teacher is an attention source over the dataset's fixed rationale, not a rationale generator).
- Reproducing a MoLSAKI-only baseline within this repository (whether to implement this separately for comparison is a decision for the experiment plan). **Decided 2026-07-14:** yes, implemented, ahead of the anchor pair's server becoming available, so the baseline is ready to run against the same teacher attention cache. No new loss math was needed -- §5's table already shows MoLSAKI-only = `L_pre + L_exp` (CODI's existing `ref_ce_loss`, computed unconditionally every forward) `+ L_att` (already implemented, gated by `use_att_loss`). What was missing was a way to turn CODI's implicit/latent path off: `CODI.forward()` (`src/model.py`) had an `if self.num_latent != 0:` guard around the latent loop already in place, but setting `num_latent=0` crashed (`logits`/`ce_loss`/`distill_loss` were only ever assigned inside that loop). Fixed by defaulting those before the loop and skipping the now-unused encoder forward when `num_latent == 0`; `num_latent > 0` behavior is byte-for-byte unchanged. New files: `scripts/train_gpt2_gsm8k-aug-nl_molsaki.sh` / `scripts/train_llama1b_gsm8k-aug-nl_molsaki.sh` (the `_isac.sh` scripts with `--num_latent 0`, and `--use_prj`/`--prj_dim`/`--prj_dropout`/`--distill_loss_div_std` dropped since they're dead in this mode), and `test_molsaki.py` (a `test.py` variant that generates directly from the tokenized question -- no `bot_id`/latent-iteration/`eot_id` -- since this training mode never touches that structure). `tests/test_train_integration.py` gained `test_molsaki_only_num_latent_zero()` covering the fix end to end.

## References
- ISAC idea.pdf (repository root)
- CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation (Shen et al., 2025, arXiv:2502.21074)
- MoLSAKI: Improving Reasoning Capabilities in Small Models through Mixture-of-Layers Distillation with Stepwise Attention on Key Information (Chen et al., EMNLP 2025) — `MoLSAKI Yao Chen.pdf` (repository root)
