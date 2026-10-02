# ISAC variant of train_llama1b_gsm8k-aug-nl.sh (ISAC.md roadmap step 8).
# Adds L_att (teacher <-> f_e stepwise attention distillation, ISAC.md Sec 2-3) on top
# of plain CODI. Requires a teacher attention cache built beforehand:
#   python cache_teacher_attention.py --teacher_model_name_or_path <TEACHER> \
#       --att_cache_dir <ATT_CACHE_DIR> [--max_examples N for a quick trial run]
#
# ISAC.md Sec 3.2 restricts the first-phase target dataset to icot-full (GSM8K-Aug-NL),
# which is why this is a variant of train_llama1b_gsm8k-aug-nl.sh specifically.
#
# Note (ISAC.md Sec 3.4/3.5): the student value-hook/MoL path was validated against a
# tiny random-init GQA Llama config (num_key_value_heads < num_attention_heads, matching
# Llama-3.2-1B-Instruct's actual architecture) -- but not yet against a real forward pass
# of this specific checkpoint. Worth a small --exp_mode True trial run first.
#
# 2026-09-11: converted to single-model multi-GPU DDP (same pattern validated on
# train_gpt2_gsm8k-aug-nl_isac.sh), data split across GPUs instead of 1. No
# train.py/model.py changes needed -- device placement already follows each tensor's
# own .device rather than a hardcoded GPU index. --nproc_per_node below must match
# the number of GPU indices passed via CUDA_VISIBLE_DEVICES exactly -- adjust it (and
# gradient_accumulation_steps, to keep per_device_train_batch_size * nproc_per_node *
# accum roughly constant at 128) if you have a different number of GPUs. On a
# dedicated 4-GPU box this is just CUDA_VISIBLE_DEVICES=0,1,2,3; on a shared server,
# check nvidia-smi (and compute_mode -- avoid anything set to Exclusive_Process) first.
# Launch with:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_llama1b_gsm8k-aug-nl_isac.sh
# If NCCL initialization hangs (accelerate prints a kernel-version warning at startup)
# or you see "CUDA-capable device(s) is/are busy or unavailable" for a GPU nvidia-smi
# showed as free, retry with NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 prefixed -- see
# ISAC_RUNBOOK.md Sec 6/10. Not needed on most fresh/modern-kernel machines; try
# without it first.
# per_device_train_batch_size stays 16 (same per-GPU memory as the single-GPU run);
# gradient_accumulation_steps is set to keep the effective batch (16 * nproc_per_node *
# accum) close to the original 128. Currently: 4 GPUs, accum 2 -> 16*4*2=128 exactly.
#
# 2026-09-29: 10000-example quick test (rented VESSL box) passed -- exp_mode flipped back
# to False for the real full-dataset run. SAVE_DIR also changed (codi_nl_llama_isac ->
# codi_nl_llama_isac_full) so this run does NOT auto-resume from the small test's
# checkpoint -- train.py resumes from get_last_checkpoint(output_dir) whenever
# output_dir/expt_name/model/epochs/lr/seed all match a prior run, which they would have
# if SAVE_DIR were left unchanged. The old test checkpoints stay at
# ~/codi_ckpt/codi_nl_llama_isac/ untouched.

SAVE_DIR=~/codi_ckpt/codi_nl_llama_isac_full

# Fill in with the teacher used to build ATT_CACHE_DIR (ISAC.md Sec 3.1/8 -- the teacher
# LLM itself is not yet decided; recorded here for provenance only, never loaded by
# train.py/CODI -- ISAC.md Sec 7.1 uses the precomputed cache instead).
TEACHER_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
# Output directory of cache_teacher_attention.py for TEACHER_MODEL_NAME_OR_PATH above.
ATT_CACHE_DIR=~/att_cache/qwen25_7b

mkdir -p "$SAVE_DIR"

cp scripts/train_llama1b_gsm8k-aug-nl_isac.sh "$SAVE_DIR"

torchrun --nproc_per_node=4 --master_port=29502 train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name gsm8k_llama1b_isac \
	--logging_dir "$SAVE_DIR/logs"\
	--logging_steps 10 \
	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
	--data_name icot-full \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 16 \
  	--gradient_accumulation_steps 2 \
	--ddp_find_unused_parameters True \
	--bf16 \
	--num_train_epochs 3 \
	--learning_rate 8e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "steps" \
	--save_steps 100 \
	--save_total_limit 2 \
	  --save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
   --num_latent 6 \
   --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 2048 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 10000 \
	--remove_eos True \
	--distill_loss_factor 20 \
	--ce_loss_factor 0.5 \
	--print_ref_model_stats True \
	--max_token_num 256 \
	--use_student_mol True \
	--use_att_loss True \
	--teacher_model_name_or_path "$TEACHER_MODEL_NAME_OR_PATH" \
	--att_cache_dir "$ATT_CACHE_DIR" \
	--att_loss_factor 50 \
	--mol_tau_teacher 0.1 \
	--mol_tau_student 0.5 \
	--critical_token_mode numeric \
