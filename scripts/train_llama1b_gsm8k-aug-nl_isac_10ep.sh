# 2026-10-01: copy of train_llama1b_gsm8k-aug-nl_isac.sh for the "10-epoch, per-epoch
# eval" experiment. Does NOT modify the original script. Same diff pattern as
# train_llama1b_gsm8k-aug-nl_10ep.sh: epochs 3->10, save_strategy steps->epoch,
# save_total_limit 2->11, SAVE_DIR suffixed _10ep. Everything else (ce_loss_factor=0.5,
# distill_loss_factor=20, att_loss_factor=50, teacher cache path, MoL taus) unchanged from
# train_llama1b_gsm8k-aug-nl_isac.sh as of 2026-10-01.
#
# Launch with: CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_llama1b_gsm8k-aug-nl_isac_10ep.sh

SAVE_DIR=~/codi_ckpt/codi_nl_llama_isac_10ep

TEACHER_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
ATT_CACHE_DIR=~/att_cache/qwen25_7b

mkdir -p "$SAVE_DIR"

cp scripts/train_llama1b_gsm8k-aug-nl_isac_10ep.sh "$SAVE_DIR"

torchrun --nproc_per_node=4 --master_port=29512 train.py \
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
