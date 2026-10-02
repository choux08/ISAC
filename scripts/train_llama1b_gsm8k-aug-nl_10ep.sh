# 2026-10-01: copy of train_llama1b_gsm8k-aug-nl.sh (CODI-only baseline) for the new
# "10-epoch, per-epoch eval" experiment. Does NOT modify the original script. Differences
# from the original:
#   - --num_train_epochs 3 -> 10 (CODI paper Appendix A's recommendation for Llama-1b;
#     the original 3-epoch run matched the original authors' own script value-for-value but
#     scored far below the paper's reported 49.7%, so this tests whether more epochs closes
#     that gap)
#   - --save_strategy "steps"/--save_steps 100 -> "epoch" (checkpoint saved at the end of
#     every epoch, not every 100 steps) so scripts/eval_per_epoch.sh can evaluate each
#     epoch's checkpoint separately and plot an accuracy-vs-epoch curve
#   - --save_total_limit 2 -> 11 (keep all 10 epoch checkpoints + headroom, instead of only
#     the last 2 -- the whole point here is comparing every epoch, not just the final one)
#   - SAVE_DIR changed (codi_nl_llama -> codi_nl_llama_10ep) so this is a genuinely fresh
#     run, not an accidental resume into the existing 3-epoch/ep_3 checkpoint (train.py's
#     output_dir already includes ep_{epochs} so ep_10 would be a distinct path anyway, but
#     a separate SAVE_DIR keeps the two experiments cleanly apart on disk)
# Everything else (lr, batch size, ce_loss_factor=0.5, distill_loss_factor=20, LoRA config,
# data) is unchanged from train_llama1b_gsm8k-aug-nl.sh as of 2026-10-01 (includes the
# ce_loss_factor=0.5 weight added that day).
#
# Launch with: CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_llama1b_gsm8k-aug-nl_10ep.sh

SAVE_DIR=~/codi_ckpt/codi_nl_llama_10ep

mkdir -p "$SAVE_DIR"

cp scripts/train_llama1b_gsm8k-aug-nl_10ep.sh "$SAVE_DIR"

torchrun --nproc_per_node=2 --master_port=29513 train.py \
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
	--num_train_epochs 10 \
	--learning_rate 8e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "epoch" \
	--save_total_limit 11 \
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
