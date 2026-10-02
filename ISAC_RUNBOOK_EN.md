## CODI + ISAC/MoLSAKI Experiment Runbook

This document covers everything from building this repo (CODI base + ISAC extension) from scratch, to running training (single GPU / multi-GPU DDP), testing, and what to check when something breaks. For architecture/math/design rationale see `CLAUDE.md` and `ISAC.md`; this document focuses on "what to actually type in the terminal."

**This document is written to be environment-agnostic** — it doesn't assume any specific server or account. Substitute the following placeholders with your own values:

| Placeholder | Meaning | Example |
|---|---|---|
| `<SERVER_HOST>` | Remote GPU server address | `cs.gs.hs.kr`, `123.45.67.89`, `myserver.example.com` |
| `<SERVER_USER>` | Server account name | `gs24110` |
| `<SERVER_REPO_PATH>` | Absolute (or home-relative) path to this repo on the server | `/src/gs24110`, `~/CODI` |
| `<LOCAL_REPO_PATH>` | Path to this repo on your local (Mac/Linux) machine | `~/Downloads/ISAC_implementation` |
| `<CONDA_ENV>` | conda environment name (pick anything) | `codi` |
| `<GPU_IDX>` | An actually-free GPU index (check with `nvidia-smi`) | `0`, `3`, `3,4,5,6` |

If you're working directly on a local machine rather than a remote server, skip sections 1–2's SSH/rsync steps and just run everything else in your local shell.

---

### 1. Building the environment

If using a remote server, connect first:

```bash
ssh <SERVER_USER>@<SERVER_HOST>
```

Then (identical whether on a server or local machine):

```bash
conda create --name <CONDA_ENV> python=3.12
conda activate <CONDA_ENV>
cd <SERVER_REPO_PATH>      # or this repo's local path, if working locally
pip install -r requirements.txt
```

HuggingFace auth (required if you use a gated model like Llama-3.2-1B-Instruct):

```bash
huggingface-cli login
```
Paste a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). You also need to accept the license on any gated model's page beforehand (e.g. `meta-llama/Llama-3.2-1B-Instruct` — this is account-based, so doing it once is enough).

Check GPUs:

```bash
nvidia-smi
```

If this is a shared multi-user server, find a GPU index with near-zero `Memory-Usage` and use only that one (`CUDA_VISIBLE_DEVICES=<GPU_IDX>`). Grabbing a GPU someone else is using will kill their job or cause OOM on yours. On a single-user machine this check matters less, but if there's more than one GPU, explicitly specifying which one to use is still good practice — HF `Trainer`'s automatic GPU detection can behave unexpectedly otherwise (see section 5).

---

### 2. Syncing files between local and server

Only relevant if you're using a remote server. **Always run this from a fresh local terminal (not from inside a session on the server):**

```bash
rsync -av <LOCAL_REPO_PATH>/ <SERVER_USER>@<SERVER_HOST>:<SERVER_REPO_PATH>/
```

To sync a single file (e.g. after only editing `train.py`):

```bash
rsync -av <LOCAL_REPO_PATH>/train.py <SERVER_USER>@<SERVER_HOST>:<SERVER_REPO_PATH>/train.py
```

Files under `src/` need the path preserved:

```bash
rsync -av <LOCAL_REPO_PATH>/src/model.py <SERVER_USER>@<SERVER_HOST>:<SERVER_REPO_PATH>/src/model.py
```

To pull results (logs, PNGs, etc.) back to your local machine, go the other direction:

```bash
scp <SERVER_USER>@<SERVER_HOST>:~/codi_ckpt/<path>/loss_plot.png ~/Downloads/
```

If you're working directly on your local machine, this entire section is unnecessary — just edit and run the files in place.

---

### 3. Building the ISAC teacher-attention cache (one-time)

ISAC/MoLSAKI training (`use_att_loss True`) requires a pre-built cache of a teacher model's attention in `att_cache_dir`. Skip this step if the cache already exists. If you need to build one (the example below uses Qwen2.5-7B-Instruct as the teacher — substitute whatever teacher model your own `ISAC.md`/experiment plan calls for):

```bash
# small trial
python cache_teacher_attention.py \
    --teacher_model_name_or_path <TEACHER_MODEL_NAME> \
    --att_cache_dir <ATT_CACHE_DIR> \
    --max_examples 50

# check meta.json (num_cached should be > 0 and close to 50)
cat <ATT_CACHE_DIR>/meta.json

# if that looks fine, do the full build (same command, drop --max_examples)
python cache_teacher_attention.py \
    --teacher_model_name_or_path <TEACHER_MODEL_NAME> \
    --att_cache_dir <ATT_CACHE_DIR>
```

`<TEACHER_MODEL_NAME>` is a HuggingFace model ID (e.g. `Qwen/Qwen2.5-7B-Instruct`), and `<ATT_CACHE_DIR>` is any path to store the cache (e.g. `~/att_cache/<teacher-name>`). If the teacher model is large (7B+), a local GPU may not have enough memory — in that case run just this step on a machine with more VRAM.

Always check cache coverage with:

```bash
cat <ATT_CACHE_DIR>/meta.json
```
Confirm `num_cached` is close to the size of the training set you actually plan to use. If it's small (only a trial run was ever done), training will skip a lot of `L_att` examples for missing cache entries (see 7-D below).

This cache is reusable regardless of `num_latent` (i.e. ISAC vs. MoLSAKI-only) — if you're running multiple experiments against the same teacher (different student models, ISAC vs. MoLSAKI, etc.), build `att_cache_dir` once and reuse it across all of them.

---

### 4. The training scripts

Under `scripts/` you'll find GSM8K-Aug-NL (`icot-full`) training scripts (ISAC/MoLSAKI only support this dataset per ISAC.md §3.2). This repo ships with four combinations by default (2 models × 2 configs):

| Script | Model | Config |
|---|---|---|
| `train_gpt2_gsm8k-aug-nl_isac.sh` | gpt2 | ISAC (`num_latent 6`) |
| `train_llama1b_gsm8k-aug-nl_isac.sh` | Llama-3.2-1B-Instruct | ISAC (`num_latent 6`) |
| `train_gpt2_gsm8k-aug-nl_molsaki.sh` | gpt2 | MoLSAKI-only baseline (`num_latent 0`) |
| `train_llama1b_gsm8k-aug-nl_molsaki.sh` | Llama-3.2-1B-Instruct | MoLSAKI-only baseline (`num_latent 0`) |

Each script defines its checkpoint save location at the top via `SAVE_DIR=...` (open the script to check).

These four are different models/configs by design, meant for **comparison** — they cannot be merged into a single training run (the underlying model architectures differ). If you have multiple GPUs, training one model split across several GPUs (DDP, section 6) and running several different experiments in parallel one-per-GPU (section 5) are two separate choices — pick whichever matches what you're trying to do.

---

### 5. Running training — Method A: one GPU per experiment, in parallel

Use this when you want to run several experiments (different models/configs) at once, each on its own GPU. Always check free GPU indices with `nvidia-smi` first and substitute accordingly (the `<GPU_IDX_N>` placeholders below are just examples — replace with real free indices):

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDX_1> nohup bash scripts/train_gpt2_gsm8k-aug-nl_isac.sh > /tmp/gpt2_isac.log 2>&1 &
disown
sleep 5
CUDA_VISIBLE_DEVICES=<GPU_IDX_2> nohup bash scripts/train_llama1b_gsm8k-aug-nl_isac.sh > /tmp/llama_isac.log 2>&1 &
disown
sleep 5
CUDA_VISIBLE_DEVICES=<GPU_IDX_3> nohup bash scripts/train_gpt2_gsm8k-aug-nl_molsaki.sh > /tmp/gpt2_molsaki.log 2>&1 &
disown
sleep 5
CUDA_VISIBLE_DEVICES=<GPU_IDX_4> nohup bash scripts/train_llama1b_gsm8k-aug-nl_molsaki.sh > /tmp/llama_molsaki.log 2>&1 &
disown
```

If you only have one GPU, don't launch all of these at once — run them one at a time, each after the previous one finishes.

**Things to watch out for**:
- When backgrounding with `&`, prefer plain `>` redirection over `tee` — piping through `tee` in the background can die along with the pipe if the terminal session drops.
- Launching several with no gap between them can cause them to die silently (log file stays empty, only an exit code shows up) due to HF Hub cache-lock / model-download contention. Put a `sleep 5` or so between each launch.
- If you omit `CUDA_VISIBLE_DEVICES`, HF `Trainer` will try to auto-wrap the model across every GPU visible on the machine with `nn.DataParallel`. On a multi-GPU shared server this was directly observed to cause `RuntimeError: CUDA error: peer mapping resources exhausted` (some driver/kernel combinations hit a peer-mapping limit when one process tries to claim many GPUs). A single- or dual-GPU personal machine may never hit this, but **explicitly setting `CUDA_VISIBLE_DEVICES` is good practice in any environment.**

---

### 6. Running training — Method B: multi-GPU DDP for a single model

For "one model, data split across several GPUs" (distributed data parallel, DDP). Any training script can be converted to this mode — you only need to change the launch command inside the script to use `torchrun`. `train.py`/`src/model.py` never hardcode a device; every tensor placement follows that tensor's own `.device`, so HF `Trainer`'s standard `torchrun`-based DDP works with no other code changes.

To convert a script to DDP:
1. Change the launch line from `python train.py \` to `torchrun --nproc_per_node=<num_gpus> --master_port=<any free port> train.py \`.
2. Divide `--gradient_accumulation_steps` by the GPU count so the effective batch (`per_device_train_batch_size × num_gpus × gradient_accumulation_steps`) stays the same as before. E.g. accum 8 on 1 GPU → accum 2 on 4 GPUs.
3. Add `--ddp_find_unused_parameters True` (needed because modules like StudentMoL aren't used on every batch, which otherwise trips a DDP error).

Run it:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDX_LIST> nohup bash scripts/<your modified script>.sh > /tmp/train_ddp.log 2>&1 &
disown
```
`<GPU_IDX_LIST>` is a comma-separated list of GPU indices (e.g. `0,1,2,3`) — it must match the count you passed to `--nproc_per_node`.

**On servers with an older kernel, NCCL initialization can hang.** If `accelerate` prints a warning at startup like `Detected kernel version X.X.X, which is below the recommended minimum of 5.5.0; this can cause the process to hang`, that hang can genuinely happen (this was directly observed on the server this project was developed on). If you see that warning, or a DDP run just stalls for no apparent reason:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=<GPU_IDX_LIST> nohup bash scripts/<your modified script>.sh > /tmp/train_ddp.log 2>&1 &
disown
```
Check your kernel version with `uname -r`. On a newer kernel/driver where this warning never appears, DDP should work fine without these flags — try without them first, and add them only if you observe a hang.

---

### 7. Changing training hyperparameters

All of these are CLI args inside `scripts/train_*.sh`, listed after `python train.py \` (or `torchrun ... train.py \`). Order doesn't matter (`HfArgumentParser` just needs them to parse).

#### 7-A. Changing the amount of training data

```
--exp_mode True \
--exp_data_num 10000 \
```
`exp_mode False` uses the entire dataset. `exp_mode True` only uses `exp_data_num` examples (for quick validation). **Important**: GSM8K-Aug / GSM8K-Aug-NL are built by augmenting each base question into ~50 contiguous variants — if you subsample by taking a front slice, you'd only see a handful of base questions repeated many times (overfitting risk). This repo's `train.py` (`SupervisedDataset.__init__`, fixed 2026-08-28) already strides evenly across the full dataset, so just changing `exp_data_num` automatically samples broadly. No further action needed. (If you swap in a different dataset, check whether it has similar augmentation structure — the fix assumes contiguous near-duplicates.)

To change the number:
```bash
sed -i 's/--exp_data_num 2000/--exp_data_num 10000/' scripts/<script name>.sh
```
or directly with an editor (vim example):
```bash
vim scripts/<script name>.sh
# search /exp_data_num (Enter) -> cw to delete the number and type the new one -> Esc -> :wq
```

#### 7-B. How total step count is computed

```
steps_per_epoch = ceil(exp_data_num / effective_batch)
total_steps = steps_per_epoch * num_train_epochs
effective_batch = per_device_train_batch_size * num_gpus(if DDP) * gradient_accumulation_steps
```
Example: `exp_data_num 10000`, `batch 16`, `accum 8`, 1 GPU, `epochs 40` → effective batch 128 → 79 steps/epoch → 3160 total steps.

To increase the total step count, raise `--num_train_epochs`, lower `--gradient_accumulation_steps` (needs memory headroom), or raise `--exp_data_num`.

#### 7-C. Batch size / memory

```
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 8 \
```
If you hit OOM, lower `per_device_train_batch_size` and raise `gradient_accumulation_steps` to keep the effective batch constant (e.g. 64/2 → 16/8). ISAC's `L_att` path runs an extra per-example forward pass for every example in the batch (when `use_att_loss True`), so memory scales roughly linearly with `per_device_train_batch_size` — this is the first thing to suspect on OOM. Reasonable values depend heavily on your GPU's memory capacity, so start small relative to your available VRAM (`nvidia-smi`'s `Memory-Usage` vs. total capacity) and increase from there.

#### 7-D. ISAC/MoLSAKI-specific flags

```
--use_student_mol True \       # register the StudentMoL module (required for use_att_loss)
--use_att_loss True \          # turn on L_att (the MoLSAKI loss)
--teacher_model_name_or_path <TEACHER_MODEL_NAME> \   # provenance only, never actually loaded
--att_cache_dir <ATT_CACHE_DIR> \                     # cache path from section 3
--att_loss_factor 1.0 \        # c_att (β), the L_att weight
--mol_tau_teacher 0.1 \        # τ1
--mol_tau_student 0.5 \        # τ2
--critical_token_mode numeric \  # only "numeric" is implemented; "keyword" is not
```
If `use_att_loss True` but `use_student_mol False` or `att_cache_dir` is missing, `CODI.__init__` raises a `ValueError` immediately (this is intentional).

`num_latent 0` means the MoLSAKI-only baseline (CODI's implicit/latent path is fully disabled). `num_latent 6` means the standard ISAC setup (CODI plus `L_att`) — the number can be changed to however many latent tokens you want.

#### 7-E. DDP-specific

```
--ddp_find_unused_parameters True \
```
Only matters when launched via `torchrun`. Harmless to leave in for single-GPU runs too (ignored).

---

### 8. Testing / evaluation

**Note**: checkpoints trained with `num_latent 0` (MoLSAKI-only) must be evaluated with `test_molsaki.py`, not `test.py` — they never learned the `bot_id`/latent/`eot_id` structure, so `test.py` would evaluate them incorrectly.

#### 8-A. ISAC checkpoints (`num_latent 6`) → `test.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDX> python test.py \
    --data_name "gsm8k" \
    --output_dir "./outputs" \
    --model_name_or_path gpt2 \
    --seed 11 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --batch_size 128 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --inf_latent_iterations 6 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir <absolute path to that experiment's checkpoint-N> \
    2>&1 | tee /tmp/test_run.log
```
`--model_name_or_path`/`--prj_dim` must match whatever model that checkpoint was trained with (e.g. for Llama1b: `--model_name_or_path meta-llama/Llama-3.2-1B-Instruct`, `--prj_dim 2048`).

`--ckpt_dir` must point at a `checkpoint-N` directory that actually exists — check first with:
```bash
ls <SAVE_DIR>/<expt_name>/<model_name>/ep_<epochs>/lr_<lr>/seed_<seed>/
```
(see section 11 for the path pattern)

You can change `--data_name` to `svamp`, `gsm-hard`, or `multi-arith` (OOD math benchmarks) for other evaluations. Raising `--inf_num_iterations` re-runs and averages accuracy over multiple samples (useful since default decoding is sampled, not greedy). With `--greedy True` the output is deterministic, so one run is enough.

#### 8-B. MoLSAKI-only checkpoints (`num_latent 0`) → `test_molsaki.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDX> python test_molsaki.py \
    --data_name "gsm8k" \
    --output_dir "./outputs" \
    --model_name_or_path gpt2 \
    --seed 11 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --batch_size 128 \
    --greedy True \
    --num_latent 0 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir <absolute path to that experiment's checkpoint-N> \
    2>&1 | tee /tmp/test_molsaki_run.log
```
CLI args are almost the same as `test.py`, minus the latent-related ones (`--use_prj`/`--prj_dim`, etc.) which don't apply when `num_latent 0`.

#### 8-C. Reading the output

```
adapter: None | GSM8K test accuracy: 4.70% |
average length of COT: 6.260803639120546
Average accuracy over 1 sampling: 4.700530705079606
```
`GSM8K test accuracy` is the number that matters. Low accuracy (single digits) is expected when training data is limited to a few thousand examples — that's a data-volume issue, not a pipeline bug. For a paper-quality comparison, you need a full run with `exp_mode False` (the entire dataset).

---

### 9. Monitoring progress

#### 9-A. Live log

```bash
tail -f /tmp/train_ddp.log
```
You should see, in order: `[CODI.init] ...` (model construction), `[train] ...` (dataset/Trainer setup), `[forward] ...` (per-step stages: student encoder pass, teacher pass, ISAC L_att pass, latent steps, computing `ref_ce_loss`, etc.), `loss=..., ce_loss=..., ref_ce_loss=..., att_loss_total=...` (every `logging_steps`), and a tqdm progress bar (`N/total_steps [elapsed<remaining, s/it]`). If all of this shows up, training is progressing normally.

Under DDP (`torchrun`, multiple GPUs), only rank 0 (the first GPU) prints these — the log is not duplicated per GPU.

#### 9-B. GPU usage

```bash
nvidia-smi
# or a lighter, targeted query
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -i <GPU_IDX_LIST>
```
With DDP across multiple GPUs, memory/utilization should be roughly even across all of them. If one GPU is way off from the others, the data isn't being split evenly.

#### 9-C. Checking whether the process is still alive

```bash
jobs -l                    # if launched with & in the current shell
ps aux | grep train.py
ps aux | grep torchrun
```

#### 9-D. Loss curves (CSV / plot)

`LossHistoryLogger` in `src/debug_utils.py` automatically writes into the same folder as the checkpoints:
```bash
tail -20 <SAVE_DIR>/.../seed_<seed>/loss_history.csv
```
A plot (PNG) is regenerated periodically at `loss_plot.png` in the same folder. To pull it from a remote server:
```bash
scp <SERVER_USER>@<SERVER_HOST>:<SAVE_DIR>/.../seed_<seed>/loss_plot.png ~/Downloads/
```
If only the CSV exists but not the PNG, matplotlib isn't installed — `pip install matplotlib`, then regenerate the plot from the existing CSV without re-running training:
```bash
python plot_loss_history.py <path to loss_history.csv> -o loss_plot.png
```

#### 9-E. tensorboard

Each script is set up with `--report_to tensorboard --logging_dir $SAVE_DIR/logs`:
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python tensorboard --logdir <SAVE_DIR>/logs
```
If your environment has a protobuf version conflict, running without that environment variable can throw `TypeError: Descriptors cannot be created directly` — add it if you hit that error. If working on a remote server, forward the port to view it in a browser:
```bash
# from a fresh local terminal
ssh -L 6006:localhost:6006 <SERVER_USER>@<SERVER_HOST>
# run tensorboard inside that session, then open localhost:6006 in your local browser
```

---

### 10. Common errors and how to diagnose them

| Symptom | Cause | Check / Fix |
|---|---|---|
| `RuntimeError: CUDA error: peer mapping resources exhausted` | Ran without `CUDA_VISIBLE_DEVICES`, so `Trainer` tries to wrap every visible GPU in `DataParallel` | Always pin `CUDA_VISIBLE_DEVICES=<GPU_IDX>` |
| `torch.OutOfMemoryError` (often around `ref_ce_loss` computation) | `per_device_train_batch_size` too large (ISAC's `L_att` per-example forward loop scales memory with batch size) | Lower the batch size, raise `gradient_accumulation_steps` to keep the effective batch the same |
| `RuntimeError: ... found at least two devices, cpu and cuda:0` (near attention_loss.py) | Missing `device=` on `compute_teacher_mol_weights`'s fallback tensor (already fixed — just confirm you're running the latest file) | Check that `src/attention_loss.py` is up to date (re-sync if on a remote server) |
| Log file is empty and a backgrounded job exits immediately | Launching multiple training runs with no gap causes a race on HF cache/model download | Run one in the foreground alone to see the real error; when launching several, space them out with `sleep` |
| `nvidia-smi` doesn't respond and the terminal seems to accept no commands | Almost always because a foreground training process is still holding the shell — what you typed was never actually executed as a command | Press `Ctrl+C` to interrupt the foreground process → confirm the prompt returns → `ps aux \| grep train.py` for orphaned processes, `kill -9` if any remain |
| Under DDP (`torchrun`), nothing prints after `[train] calling trainer.train()...` and it just sits there, with `accelerate.utils.other:Detected kernel version ... can cause the process to hang` above it | Your kernel is below accelerate's recommended minimum (5.5.0), causing a hang during NCCL initialization | Retry with `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` prefixed to the launch command (see section 6). Check kernel version with `uname -r` |
| `FileNotFoundError: ... checkpoint-N/model.safetensors` (or `pytorch_model.bin`) in test.py/test_molsaki.py | That checkpoint directory was deleted, or training never reached that step | `ls <SAVE_DIR>/.../seed_<seed>/` to see which `checkpoint-N` actually exists, then fix `--ckpt_dir` |
| `ConnectionResetError` (test.py, while contacting the HF Hub) | Transient network issue | Retry. If persistent, check connectivity with `curl -I https://huggingface.co` |
| tensorboard `TypeError: Descriptors cannot be created directly` | protobuf version conflict | Run with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` prefixed |
| In tmux, scrolling doesn't move the view and instead replays old commands | Mouse mode is off | Add `set -g mouse on` to `~/.tmux.conf` and restart tmux; in a pinch, `Ctrl+b [` enters copy-mode |
| `use_att_loss True` but `L_att` stays near zero or a lot of examples are skipped | `att_cache_dir` doesn't cover the range of data currently being sampled (especially with `exp_data_num`'s wide-stride sampling) | Check `num_cached` in `cat <att_cache_dir>/meta.json` against the full dataset size, and check `att_loss_num_skipped_no_cache` in the log |
| Re-running without deleting `SAVE_DIR` continues from the old run instead of starting fresh | `train.py` auto-resumes from the latest `checkpoint-N` whenever `output_dir` already exists (intentional feature) | If you want a genuine fresh start, always `rm -rf <SAVE_DIR>` first |

General diagnostic order: **1) check the Traceback at the bottom of the log → 2) note which file/line it's from → 3) look for a matching symptom in the table above → 4) if nothing matches, start with `nvidia-smi`/`ps aux` to check process/GPU state.**

---

### 11. Checkpoints and resuming

- Save path pattern: `{output_dir}/{expt_name}/{model_name}/ep_{epochs}/lr_{lr}/seed_{seed}/checkpoint-{step}` (`output_dir` is each script's `SAVE_DIR`; the rest are filled in automatically from the CLI args).
- `--save_strategy steps --save_steps 500 --save_total_limit 2`: saves every 500 steps, keeping only the latest 2 (older ones are deleted automatically). Adjust `--save_steps` to whatever save frequency you want.
- Re-running a script with the same `SAVE_DIR` automatically resumes from the latest `checkpoint-N` (model, optimizer, scheduler, RNG, and step count are all restored). If the process dies mid-run, just re-run the same command.
- **To start completely fresh (e.g. after changing a setting), always delete first:**
  ```bash
  rm -rf <SAVE_DIR>
  ```
  (path differs per experiment — check each script's `SAVE_DIR=...` at the top)

---

### 12. Experiment design reference

Teacher/student model pairings, comparison baselines (e.g. CODI-only / MoLSAKI-only / ISAC), and similar concrete experiment-plan details belong in this repo's `CLAUDE.md`/`ISAC.md` (or your own project's documentation) if you've written one up. This document only covers "how to run it" — "what to actually experiment with" is project-specific, so it's worth keeping that as a separate document.
