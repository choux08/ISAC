# ISAC variant of train_gpt2_gsm8k-aug-nl.sh (ISAC.md roadmap step 8).
# Adds L_att (teacher <-> f_e stepwise attention distillation, ISAC.md Sec 2-3) on top
# of plain CODI. Requires a teacher attention cache built beforehand:
#   python cache_teacher_attention.py --teacher_model_name_or_path <TEACHER> \
#       --att_cache_dir <ATT_CACHE_DIR> [--max_examples N for a quick trial run]
#
# ISAC.md Sec 3.2 restricts the first-phase target dataset to icot-full (GSM8K-Aug-NL),
# which is why this is a variant of train_gpt2_gsm8k-aug-nl.sh specifically (not the
# word-split icot/commonsense/strategy variants -- those aren't ISAC-supported yet).
#
# 2026-08-28: switched to single-model 4-GPU DDP (data split across 4 GPUs, one set of
# weights) instead of 4 independent single-GPU experiments, per user request to cut
# memory/runtime. No train.py/model.py changes needed -- device placement there already
# follows each tensor's own .device rather than a hardcoded GPU index, so HF Trainer's
# standard torchrun-based DDP just works. --nproc_per_node below must match the number
# of GPU indices passed via CUDA_VISIBLE_DEVICES exactly. On a dedicated 4-GPU box this
# is just CUDA_VISIBLE_DEVICES=0,1,2,3; on a shared server, check nvidia-smi first and
# use whichever 4 indices are actually free. Launch with:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_gpt2_gsm8k-aug-nl_isac.sh
# per_device_train_batch_size stays 16 (same per-GPU memory as the old single-GPU run);
# gradient_accumulation_steps dropped 8->2 so the *effective* batch stays 16*2*4=128
# (same as before) but is now reached ~4x faster since 4 GPUs process it in parallel
# instead of 8 sequential accumulation micro-steps on 1 GPU.
# If NCCL initialization hangs (accelerate prints a kernel-version warning at startup),
# retry with NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 prefixed -- see ISAC_RUNBOOK.md Sec 6.
# Not needed on most fresh/modern-kernel machines; try without it first.

SAVE_DIR=~/codi_ckpt/codi_nl_gpt2_isac

# Fill in with the teacher used to build ATT_CACHE_DIR (ISAC.md Sec 3.1/8 -- the teacher
# LLM itself is not yet decided; recorded here for provenance only, never loaded by
# train.py/CODI -- ISAC.md Sec 7.1 uses the precomputed cache instead).
TEACHER_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
# Output directory of cache_teacher_attention.py for TEACHER_MODEL_NAME_OR_PATH above.
ATT_CACHE_DIR=~/att_cache/qwen25_7b

mkdir -p "$SAVE_DIR"

cp scripts/train_gpt2_gsm8k-aug-nl_isac.sh "$SAVE_DIR"

torchrun --nproc_per_node=4 --master_port=29501 train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name gsm8k_gpt2_isac \
	--logging_dir "$SAVE_DIR/logs"\
	--logging_steps 10 \
	--model_name_or_path gpt2 \
	--data_name icot-full \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 16 \
  	--gradient_accumulation_steps 2 \
	--ddp_find_unused_parameters True \
	--bf16 \
	--num_train_epochs 40 \
	--learning_rate 3e-3 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "steps" \
	--save_steps 500 \
	--save_safetensors False \
	--save_total_limit 2 \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
    --num_latent 6 \
    --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 768 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode True \
	--exp_data_num 10000 \
	--remove_eos True \
	--print_ref_model_stats True \
	--use_student_mol True \
	--use_att_loss True \
	--teacher_model_name_or_path "$TEACHER_MODEL_NAME_OR_PATH" \
	--att_cache_dir "$ATT_CACHE_DIR" \
	--att_loss_factor 1.0 \
	--mol_tau_teacher 0.1 \
	--mol_tau_student 0.5 \
	--critical_token_mode numeric \
