# 2026-09-30: 2-GPU DDP (rented instance has 4 GPUs total, 0-3), so this and the
# MoLSAKI-only baseline can run side-by-side at the same time, each pinned to a disjoint
# half. gradient_accumulation_steps raised 2 -> 4 to keep the effective batch at
# 16*2*4=128, same as the 4-GPU version -- only wall-clock time is affected. This is the
# CODI-only comparison baseline for the ISAC full run -- same epochs (3), same lr, same
# full dataset (exp_mode False). --save_steps stays 100 for crash-resilience
# (ISAC_RUNBOOK.md Sec 6/10).
# Launch with: CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_llama1b_gsm8k-aug-nl.sh
# (run the MoLSAKI-only script at the same time with CUDA_VISIBLE_DEVICES=2,3 -- the two
# scripts already use different --master_port values so they won't collide.)

SAVE_DIR=~/codi_ckpt/codi_nl_llama

mkdir -p "$SAVE_DIR"

cp scripts/train_llama1b_gsm8k-aug-nl.sh "$SAVE_DIR"

torchrun --nproc_per_node=2 --master_port=29503 train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name gsm8k_llama1b_latent_baseline \
	--logging_dir "$SAVE_DIR/logs"\
	--logging_steps 10 \
	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
	--data_name icot-full \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 16 \
  	--gradient_accumulation_steps 4 \
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
	--exp_data_num 1000 \
	--remove_eos True \
	--distill_loss_factor 20 \
	--ce_loss_factor 0.5 \
	--print_ref_model_stats True \
	--max_token_num 256
