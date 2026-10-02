# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Official implementation of **CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation** (EMNLP 2025, arXiv:2502.21074). A single decoder-only LM (GPT-2 or Llama-3.2-1B-Instruct) is trained via LoRA to act as both a "teacher" (explicit natural-language CoT) and a "student" (implicit CoT reasoning through a small number of continuous latent tokens), with the student's hidden states distilled from the teacher's.

## Environment setup

```
conda create --name codi python=3.12
conda activate codi
pip install -r requirements.txt
```

There is no lint/CI setup in this repo — it's a research codebase driven entirely by the shell scripts in `scripts/`. There is a `tests/` directory (added for the ISAC extension, see below); its tests are plain `python tests/<file>.py` scripts, not a pytest/CI suite, and the end-to-end one requires a GPU + network.

## Common commands

All entry points are plain `python` scripts driven by `transformers.HfArgumentParser`, invoked via shell scripts in `scripts/`. **The scripts hard-code absolute paths from the original authors' cluster** (`/scratch/prj/...`, `/ephemeral/...`, `~/codi_ckpt/...`) — adjust `SAVE_DIR`/`--ckpt_dir` before running on this machine, don't assume a script works unmodified.

Evaluate a pretrained checkpoint on GSM8K:
```
bash scripts/test_gpt2.sh     # or scripts/test_llama1b.sh
```
`--data_name` can be swapped to `svamp`, `gsm-hard`, `multi-arith` (OOD math benchmarks) or `commonsense`.

Train from scratch:
```
bash scripts/train_gpt2_gsm8k-aug.sh          # or train_llama1b_gsm8k-aug.sh
bash scripts/train_gpt2_gsm8k-aug-nl.sh       # NL-CoT variant (data_name=icot-full)
bash scripts/train_gpt2_commonsense.sh        # or train_llama_commonsense.sh
```

Probe/visualize latent thoughts (decodes each latent token through the LM head, Section 5 of the paper):
```
bash scripts/probe_latent_token.sh
```
Output written to `outputs/decoded_latent.txt`.

There's no separate `--do_eval` step: `train.py` is train-only, `test.py`/`probe_latent_token.py` load a saved checkpoint (`model.safetensors` or `pytorch_model.bin`) from `--ckpt_dir` and evaluate. `test.py` and `probe_latent_token.py` re-run evaluation `--inf_num_iterations` times and average accuracy (since generation is sampled, not greedy, by default).

### ISAC commands (the extension in this repo — see "ISAC extension" below)

ISAC training requires a one-time offline pass that caches the external teacher LLM's attention to disk, then a normal `train.py` run with the ISAC flags pointed at that cache:
```
python cache_teacher_attention.py \
    --teacher_model_name_or_path <TEACHER> \
    --att_cache_dir <ATT_CACHE_DIR>          # add --max_examples N for a quick trial

bash scripts/train_gpt2_gsm8k-aug-nl_isac.sh    # or scripts/train_llama1b_gsm8k-aug-nl_isac.sh
```
All four ISAC/MoLSAKI-only scripts have `TEACHER_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct` and `ATT_CACHE_DIR=~/att_cache/qwen25_7b` filled in (2026-07-16, anchor pair — see "ISAC experiment plan" below); the cache itself still needs to be built there first via `cache_teacher_attention.py` before any of these scripts can run. ISAC only supports the `icot-full` (GSM8K-Aug-NL) dataset (ISAC.md §3.2).

Run the ISAC tests:
```
python tests/test_critical_tokens.py            # step/critical-token indexing
python tests/test_attention_loss.py             # MoL weights + L_att math
python tests/test_att_cache_consistency.py      # cache vs. live aggregation match
python tests/test_student_mol_and_value_hooks.py
python tests/test_train_integration.py          # SLOW: needs GPU + network (downloads gpt2 + dataset)
```

### MoLSAKI-only baseline commands (ISAC.md §5/§8, decided 2026-07-14)

Same teacher attention cache as above, but with `--num_latent 0` (disables CODI's implicit/latent path entirely, so training is just `ref_ce_loss` + `att_loss`):
```
bash scripts/train_gpt2_gsm8k-aug-nl_molsaki.sh    # or scripts/train_llama1b_gsm8k-aug-nl_molsaki.sh
```
Evaluate with `test_molsaki.py`, **not** `test.py` — this checkpoint never trains on `bot_id`/latent/`eot_id`, so it must be evaluated by generating directly from the tokenized question (same CLI args as `test.py`).

## Architecture

### Single-model teacher/student design (`src/model.py`, class `CODI`)

`CODI` wraps one `AutoModelForCausalLM` (`self.codi`), optionally LoRA-adapted, and vocab is extended by 3 tokens: `pad_token_id`, `bot_id` (begin-of-thought), `eot_id` (end-of-thought) appended after the original vocab.

`forward()` does two passes per batch:
1. **Teacher pass** (`ref_input_ids` = question + explicit CoT text + answer): a normal causal-LM forward, producing `ref_ce_loss` and per-layer hidden states used as distillation targets. Run once under `torch.no_grad()` for stable targets and once with grad for its own CE loss.
2. **Student pass** (`encoder_input_ids` = question + `bot_id`): the LM encodes the question, then autoregressively produces `num_latent` continuous "latent thought" embeddings — each step feeds the previous step's last-layer hidden state (optionally through `self.prj`, a small MLP+LayerNorm projection) back in as `inputs_embeds` for the next step, using KV-cache (`past_key_values`) rather than emitting real tokens. After the last latent, the model decodes `decoder_input_ids` (`eot_id` + NL answer) to produce the final answer, giving `ce_loss`.

The **distillation loss** aligns the student's hidden states (at the answer-generation position, every layer) with the teacher's, via SmoothL1 or L2 (`distill_loss_type`), optionally divided by the teacher's std for scale-invariance (`distill_loss_div_std`). Total loss = `ce_loss_total + distill_loss_total + ref_ce_loss + att_loss_total` (each independently weighted). The final `att_loss_total` term is the ISAC extension's `L_att` and is **`0` by default** — it is only nonzero when `use_att_loss=True` (see "ISAC extension" below), so plain CODI runs are unaffected.

**`num_latent=0`** (ISAC.md §5/§8, decided 2026-07-14) skips the student/implicit pass entirely — `ce_loss_total`/`distill_loss_total` stay `0`, so the total loss reduces to `ref_ce_loss + att_loss_total`. This is the MoLSAKI-only baseline (no implicit path to distill into, matching the original MoLSAKI paper's setup); see `scripts/train_*_molsaki.sh` and `test_molsaki.py` below. `num_latent > 0` behavior is unaffected.

`get_embd()` resolves the token-embedding table across model families (GPT-2 vs. Pythia vs. Llama/Mistral/Qwen-style), with and without a LoRA wrapper — needed anywhere embeddings must be looked up manually (e.g. feeding `eot_id`/generated-token embeddings back into the model during inference).

### Training pipeline (`train.py`)

- `CustomTrainer(Trainer)` overrides `compute_loss` to inject `step`/`step_ratio` into the forward call and log the loss components (`ce_loss`, `distill_loss`, `ref_ce_loss`) separately.
- Dataset selection is dispatched by substring match on `--data_name` inside `SupervisedDataset`/`make_supervised_data_module`:
  - `"icot" + "full"` → `zen-E/GSM8k-Aug-NL` (natural-language CoT; CoT is split on `". "` and optionally truncated one step via `include_last_cot`)
  - `"icot"` (without "full") → `zen-E/GSM8k-Aug` (symbolic/structured CoT; the CoT string is split by *word*, and progressively truncated per latent index — `cot_list` — this is legacy/unused output, current code just joins the full remaining cot)
  - `"commonsense"` / `"strategy"` → `zen-E/CommonsenseQA-GPT4omini` / `zen-E/StrategyQA_CoT_GPT4o`
  - `"prontoqa"` → reads a **hard-coded local path** (`/home/ubuntu/coconut/data/prontoqa_train.json`) — will not work outside the original author's machine
- Preprocessing builds two parallel tokenized sequences per example: `ref_input_ids`/`ref_labels` (teacher: question+CoT+answer, CE-masked over the question) and `encoder_input_ids`/`decoder_input_ids`/`labels` (student: question+`bot_id` as encoder input, `eot_id`+answer as decoder input). `get_answer_token_position` locates the position right after the literal phrase `"The answer is:"` (or `"The next step result is:"`) — this position is where distillation and CE loss are evaluated, so answer formatting in the data must contain that exact phrase.
- `DataCollatorForSupervisedDataset` left-pads `encoder_input_ids` (so generation continues naturally from the right edge) and right-pads everything else.
- Output checkpoints are written to `{output_dir}/{expt_name}/{model_name}/ep_{epochs}/lr_{lr}/seed_{seed}/`. All `scripts/train_*.sh` set `--save_strategy steps --save_steps 500 --save_total_limit 2` (mid-training checkpoints, resilient to the process dying partway through a long run — previously every script used `--save_strategy no`, so a crash meant total loss). `train.py` auto-resumes: before calling `trainer.train()`, it checks `training_args.resume_from_checkpoint` (explicit override) then falls back to `get_last_checkpoint(training_args.output_dir)` — so simply re-running the same script with the same `output_dir`/`expt_name`/model/epochs/lr/seed (i.e. the same resolved output_dir) picks up from the latest `checkpoint-N` (model, optimizer, scheduler, RNG, and step count all restored) instead of starting over. Verified end-to-end on this repo's `CODI` wrapper (a plain `nn.Module`, not a `PreTrainedModel`) since this codepath was never previously exercised here. `--save_steps 500` is a starting point — tune it once real per-step throughput on the actual training server is known.

### Evaluation (`test.py`, `probe_latent_token.py`)

Both scripts reimplement generation manually (no `model.generate()`) because inference alternates between real tokens and continuous latent embeddings: encode question → `inf_latent_iterations` latent steps (feeding the last hidden state back as `inputs_embeds`, same as training) → feed `eot_id` (+ eos) embedding → autoregressive top-k/top-p sampling (or greedy if `--greedy True`) over real tokens until EOS or `max_new_tokens`. `extract_answer_number` pulls the last numeric literal out of the decoded text as the predicted answer; commonsense/strategy/prontoqa datasets are matched specially (letter choice / True-False).

`probe_latent_token.py` additionally runs each latent embedding through `model.codi.lm_head` before generation to get the top-5 vocab tokens the latent "would decode to" — this is what makes the latent thoughts human-interpretable, and is the only difference from `test.py`'s eval loop.

### ISAC extension code (`L_att`) — new files and hooks into existing code

The ISAC extension (roadmap steps 1–8 of `ISAC.md`, all implemented) adds a fourth loss term `L_att` on top of the three above. It is fully gated behind `use_att_loss`/`use_student_mol` (default `False`), so everything below is dormant on a plain CODI run. New/changed pieces:

- **`src/critical_tokens.py`** (new) — `get_step_and_critical_token_indices(question, rationale, tokenizer, mode)` returns the per-reasoning-step token-index sets (`M1`/`N1`, split on periods) and per-critical-token index sets (`M2`/`N2`). Only `mode="numeric"` (numeric literals as critical tokens, for GSM8K-Aug-NL) is implemented; `"keyword"` raises `NotImplementedError`. Requires a **fast** tokenizer (uses `offset_mapping`). Tokenizer-agnostic on purpose so teacher and student can be different model families (ISAC.md §3.3).
- **`src/attention_loss.py`** (new) — `aggregate_layer_attention` (sum a head-averaged `(seq,seq)` attention into a `(step, critical-token)` matrix), `compute_teacher_mol_weights` (parameter-free gradient-statistic layer weighting, `τ1`; falls back to uniform when `<2` critical tokens), `combine_layer_attention` (shared teacher/student `Σ_l p_l·A_l`), and `compute_attention_loss` (row-wise softmax → KL, per example).
- **`StudentMoL`** (new class in `src/model.py`) — trainable value-vector-based Mixture-of-Layers (RMSNorm → sum over seq → Linear → softmax `τ2`). Registered in `CODI.__init__` as `self.student_mol` only when `use_student_mol=True`, and trains jointly with the LoRA params. Sized by `CODI.get_value_dim()` (GQA-aware: value width can be `< hidden_size` for Llama/Mistral/Qwen).
- **`CODI._register_value_hooks()`** (new) — forward hooks capturing each layer's value projection `V_l` into `self._captured_values` (GPT-2 fused `c_attn` split, or `v_proj` for the Llama family; Pythia is an explicit `NotImplementedError`).
- **`CODI.forward()`** — after the `ref`/explicit pass, if `use_att_loss`, runs one **extra dedicated forward pass per example** on `question+rationale` alone (via a fast tokenizer, unpadded), captures the student's `output_attentions` + value vectors, aggregates into `A^S`, loads the cached teacher `A^L` from `att_cache_dir/{raw_index}.pt`, and adds the KL `L_att`. It does **not** reuse `ref_input_ids` (which interleaves EOS delimiters) — see ISAC.md §3.5. Needs `raw_index`/`question_texts`/`rationale_texts` threaded through the batch (added to `DataCollatorForSupervisedDataset`; HF `Trainer` passes non-tensor batch values through untouched).
- **`CODI.__init__`** — when `use_att_loss=True`, calls `_check_att_cache_consistency()`, which hard-fails if `att_cache_dir/meta.json` was built with a different `critical_token_mode`/`include_last_cot` (ISAC.md §3.6.1).
- **`cache_teacher_attention.py`** (new, repo root) — standalone offline script: loads a frozen teacher LLM, forwards each `question+rationale`, aggregates per-layer attention into raw `A_l^L`, and writes one `{raw_index}.pt` per example + a `meta.json`. **CODI never loads a teacher model itself** — it only reads this cache (ISAC.md §3.1/§7.1). MoL combination is deferred to train time (it's a parameter-free statistic, so it doesn't require redoing the teacher forward).
- **`test_molsaki.py`** (new, repo root) — a `test.py` variant for the MoLSAKI-only baseline (`num_latent=0`): generates directly from the tokenized question, no `bot_id`/latent-iteration/`eot_id` handling, since that training mode never touches CODI's encoder/decoder structure. Reuses `test.py`'s dataset loading, sampling loop, and `extract_answer_number`/`compute_accuracy` verbatim.

Key deviations from the ISAC.md spec (all documented there): the teacher is cache-only rather than loaded live (§3.1), and `L_att` uses a separate student forward pass rather than the `ref` tensor (§3.5).

## Key CLI arguments (all scripts, via `HfArgumentParser`)

Defined across `ModelArguments`, `DataArguments`, `TrainingArguments` in `src/model.py` (`TrainingArguments` subclasses `transformers.TrainingArguments`, so all standard HF training args are also available: `--num_train_epochs`, `--learning_rate`, `--lr_scheduler_type`, etc.).

- `num_latent` — number of continuous latent thoughts used during training; `inf_latent_iterations` — same, but for inference (can differ from training).
- `use_prj` / `prj_dim` / `prj_no_ln` / `prj_dropout` — optional projection MLP applied to each latent embedding before feeding it back in.
- `distill_loss_type` (`smooth_l1` or `l2`), `distill_loss_div_std`, `distill_loss_factor`, `ref_loss_factor` — control the distillation vs. teacher-CE loss balance.
- `include_last_cot` — whether the final CoT step is included in training data (excluded by default so the model must infer it).
- `remove_eos` — drop the `<eos>` delimiter between question and CoT.
- `fix_attn_mask` — a documented bug-workaround flag; leave `False` unless deliberately testing it.
- `max_token_num` — filters out training examples whose tokenized length exceeds this, to avoid OOM.
- `greedy` (test-time decoding), `inf_num_iterations` (repeat eval this many times and average, since default decoding is sampled).

### ISAC arguments (all in `TrainingArguments`, default off — plain CODI unaffected)

- `use_student_mol` — register the trainable `StudentMoL` module. Required when `use_att_loss=True`.
- `use_att_loss` — master gate for the whole `L_att` path. Requires `use_student_mol=True` **and** `att_cache_dir`.
- `att_cache_dir` — directory of per-example teacher attention produced by `cache_teacher_attention.py`; consistency-checked against its `meta.json` at construction time.
- `teacher_model_name_or_path` — provenance/metadata only; **never loaded at train time** (CODI reads the cache, not a live teacher).
- `att_loss_factor` (`c_att`/MoLSAKI's `β`, default `1.0`), `mol_tau_teacher` (`τ1`, default `0.1`), `mol_tau_student` (`τ2`, default `0.5`) — `L_att` weight and the two MoL softmax temperatures (ISAC.md §2.4 defaults).
- `critical_token_mode` — `"numeric"` (implemented, math) or `"keyword"` (not implemented). Must match the mode `att_cache_dir` was built with.

## Pretrained weights

`zen-E/CODI-gpt2` and `zen-E/CODI-llama3.2-1b-Instruct` on Hugging Face Hub — referenced via `--ckpt_dir` pointing at a local download containing `model.safetensors` or `pytorch_model.bin`, not loaded directly by HF model ID.

## ISAC extension (implemented: roadmap steps 1–8 of `ISAC.md`)

This repo is CODI **plus** the ISAC extension. **Read [`ISAC.md`](ISAC.md) before touching anything ISAC-related** — it is the authoritative spec (full math, line-level mapping onto CODI, every documented deviation, the step-by-step roadmap, and the remaining risks). This section is only a map; ISAC.md is the source of truth.

**What ISAC is:** it adds MoLSAKI's stepwise-attention distillation loss (`L_att`) between a real external teacher LLM and this model's *explicit* path (`f_e` = CODI's `ref_input_ids` pass), on top of CODI's existing self-distillation loss (`L_KD`, = `distill_loss`) between the explicit and implicit paths. Final objective `L_ISAC = c_ce,e·L_ce,e + c_ce,i·L_ce,i + c_att·L_att + c_KD·L_KD` (ISAC.md §1). Terminology: what CODI calls "teacher" (the explicit-CoT forward pass) is ISAC's `f_e`; ISAC's actual **teacher** is a separate, larger, frozen LLM used only as a source of attention patterns to distill.

**Implementation status:**
- **Done (steps 1–8):** critical-token/step indexing, teacher-attention caching script, `StudentMoL`, teacher/student MoL, `L_att` computation + `CODI.forward()` integration, all new `TrainingArguments`, ISAC training scripts, and a committed `tests/` suite. The code, files, and hooks are described under "ISAC extension code" in the Architecture section above; the new flags under "ISAC arguments"; the commands under "ISAC commands." Defaults keep plain CODI byte-for-byte unaffected.
- **Not done (steps 9–10, ISAC.md §6):** (9) a real training run confirming `L_att` actually trends *downward* over many steps (only mechanical shape/finite/gradient checks exist), and (10) accuracy evaluation vs. the CODI-only baseline to test the core hypothesis ("MoLSAKI-level accuracy at CODI-level latency," ISAC.md §5/§7.3 — **still unverified**). A standalone MoLSAKI-only baseline (needed for this comparison) is now reproducible in-repo via `--num_latent 0` — see "MoLSAKI-only baseline commands" above and ISAC.md §8's 2026-07-14 entry.
- **Blockers / open:** the teacher/student pair is **now chosen** (Qwen2.5-7B-Instruct → Llama-3.2-1B-Instruct anchor; full ranking under "ISAC experiment plan" below) and all four training scripts' placeholders are filled in (2026-07-16) — the only remaining prerequisite is actually running `cache_teacher_attention.py` on a machine with enough VRAM for Qwen2.5-7B-Instruct (this repo's local dev machine only has a 6GB GPU, insufficient — see "ISAC experiment plan" below for the server runbook). `critical_token_mode="keyword"` (commonsense) and the word-split `icot` dataset are **not** implemented (ISAC.md §7.4). The extra per-example student forward pass in `L_att` (up to `per_device_train_batch_size` extra forwards + disk loads per step) is **not yet profiled** (ISAC.md §7.1) — worth measuring before a full run.

Cross-references into ISAC.md: §2.4 for the validated defaults (`β=1.0`, `τ1=0.1`, `τ2=0.5`); §3.2–3.3 for the critical-token definition and why teacher/student tokenizer mismatch is a non-issue by construction; §3.5/§3.1 for the two main deviations (separate student forward pass; cache-only teacher); §5 for the CODI-only / MoLSAKI-only / ISAC comparison; §7 for the remaining risks.

### ISAC experiment plan & run notes (decided 2026-07-10)

**Teacher/student pairs (run in this order).** The teacher is only an *attention source* — it forwards the dataset's fixed `question+rationale` from GSM8K-Aug-NL and its attention is read; it does **not** generate rationales. So pick it for GSM8K reading quality, not tokenizer compatibility (ISAC.md §3.3); cross-family teacher↔student is fine by construction. Model IDs are given so scripts/`cache_teacher_attention.py` can be filled in directly (one `att_cache_dir` per teacher).

1. **`Qwen/Qwen2.5-7B-Instruct` → `meta-llama/Llama-3.2-1B-Instruct`** (anchor) — strongest 7–8B teacher on GSM8K (~91.6%) + the real target student.
2. **`Qwen/Qwen2.5-7B-Instruct` → `gpt2`** — hold teacher, shrink student: isolates the student-scale effect, fastest iteration, directly comparable to the CODI-gpt2 baseline and MoLSAKI's small-student regime.
3. **`meta-llama/Llama-3.1-8B-Instruct` → `meta-llama/Llama-3.2-1B-Instruct`** — hold student, swap teacher to MoLSAKI's exact 8B teacher: isolates teacher choice and, being same-family (Llama↔Llama), acts as a control that the anchor's cross-family (Qwen→Llama) setup introduces no artifacts.
- Optional ceiling test (4th): `Qwen/Qwen2.5-32B-Instruct` → `meta-llama/Llama-3.2-1B-Instruct`. Caching is offline/one-time, so the larger teacher is affordable.

The design varies one factor per step around the anchor (pair 2 = student axis, pair 3 = teacher axis) so accuracy differences are attributable.

**`attn_implementation` caveat (student's `L_att` forward).** `CODI.__init__` loads `self.codi` with no `attn_implementation`, so it defaults to **SDPA** (transformers 4.52.4 + torch 2.7). SDPA can't return attention weights, but `CODI.forward()`'s L_att student forward requests `output_attentions=True`; in 4.52.4 this **auto-falls back to eager for that one forward** (attentions returned, gradients still flow), while the main training forwards keep fast SDPA. **Do not** globally set `attn_implementation="eager"` on `self.codi` just for this — it would slow every main forward (ref pass, latent loop, decoder), none of which need attentions. Before a long run with a new student, smoke-test `model(input_ids, output_attentions=True).attentions is not None`; only if the fallback is absent on some future transformers version should the student then be loaded with eager. Value-vector capture via the `v_proj`/`c_attn` hooks is independent of the attention backend and works either way.

### Server runbook (decided 2026-07-16)

The local dev machine's GPU (6GB) can't fit `Qwen/Qwen2.5-7B-Instruct` (~15GB in bf16), so the teacher-attention caching step (and the real training runs) must happen on a separate, larger server. All four scripts' placeholders are pre-filled (`TEACHER_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct`, `ATT_CACHE_DIR=~/att_cache/qwen25_7b`) — run these steps on the server, in order:

1. **Environment**: `conda create --name codi python=3.12 && conda activate codi && pip install -r requirements.txt`.
2. **HF auth**: `huggingface-cli login` (paste a token from https://huggingface.co/settings/tokens). `meta-llama/Llama-3.2-1B-Instruct` is gated — the license must be accepted on huggingface.co under the same account (this is account-based, not machine-based, so it carries over if already accepted elsewhere).
3. **GPU sanity check**: `nvidia-smi` — confirm enough VRAM for `Qwen2.5-7B-Instruct` in bf16 (~15GB weights; `cache_teacher_attention.py` loads it with `attn_implementation="eager"`, which materializes full `(seq,seq)` attention per layer, so budget headroom beyond just the weights — ~24GB+ recommended). If VRAM is tight, say so and 4-bit loading can be added to `cache_teacher_attention.py` (not currently implemented — it loads full bf16 only).
4. **Attention smoke test** (re-verify on the new GPU/transformers build, per the caveat above): confirm `AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16)(input_ids, output_attentions=True).attentions is not None`.
5. **Trial cache run** (sanity-check the pipeline before committing to the full dataset):
   ```
   python cache_teacher_attention.py \
       --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
       --att_cache_dir ~/att_cache/qwen25_7b \
       --max_examples 50
   ```
   Check `~/att_cache/qwen25_7b/meta.json` — `num_cached` should be > 0 and roughly close to 50 (minus any skipped for missing critical tokens/length).
6. **Full cache build** (same command, drop `--max_examples`) — this is the one genuinely long/expensive step; everything after it is comparatively cheap and reruns of `cache_teacher_attention.py` for the same teacher are unnecessary once this completes (both `_isac.sh` and `_molsaki.sh` scripts, for both students, reuse this same `att_cache_dir`).
7. **Launch training** — recommended order: `bash scripts/train_gpt2_gsm8k-aug-nl_isac.sh` first (pair 2, fastest iteration — confirms the whole `L_att` pipeline works against the real Qwen2.5-7B cache before committing to the slower anchor), then `bash scripts/train_llama1b_gsm8k-aug-nl_isac.sh` (pair 1, the anchor). Run the two `_molsaki.sh` scripts (same order) for the §5/§7.3 MoLSAKI-only baseline comparison.
8. **Monitor**: `tensorboard --logdir <SAVE_DIR>/logs` (each script sets `--report_to tensorboard` and `--logging_dir $SAVE_DIR/logs`).
9. **Resilience**: if the server dies mid-run, just re-run the same script unchanged — `train.py` auto-resumes from the latest `checkpoint-N` under `SAVE_DIR` (see "Training pipeline" above; verified end-to-end on this repo's `CODI` wrapper).
10. **Evaluate**: `test.py` for the `_isac.sh` checkpoints (`num_latent>0`, implicit-path eval); `test_molsaki.py` for the `_molsaki.sh` checkpoints (`num_latent=0`, direct explicit-CoT generation) — do not cross the two, the checkpoints are trained for different eval paths.

## Editing Rules

Unless explicitly requested, do not:

- rename existing classes
- reorganize directories
- rewrite working implementations
- change inference behavior
- modify latent generation logic
- change data preprocessing

When implementing a new feature,
prefer extending existing code instead of replacing it.

## Before Modifying Code

Before editing any file:

1.
Explain why the file needs modification.

2.
Identify the corresponding section of ISAC.md.

3.
Identify the corresponding implementation in CODI.

4.
Describe only the required differences.

Only then implement the change.

