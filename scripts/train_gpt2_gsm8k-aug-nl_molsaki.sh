# MoLSAKI-only baseline variant of train_gpt2_gsm8k-aug-nl_isac.sh (ISAC.md Sec 5/8,
# decided 2026-07-14). Sets --num_latent 0 to disable CODI's implicit/latent path
# entirely (src/model.py's CODI.forward() now supports this -- see the num_latent == 0
# guards added there), so training reduces to exactly MoLSAKI's two loss terms:
#   ref_ce_loss (L_pre + L_exp, the explicit-CoT CE loss CODI already computes
#   unconditionally) + att_loss_total (L_att, unchanged, still gated by use_att_loss).
# There is no L_ce,i or L_KD in this mode (nothing to distill into -- no implicit path).
#
# Requires the same teacher attention cache as the ISAC script:
#   python cache_teacher_attention.py --teacher_model_name_or_path <TEACHER> \
#       --att_cache_dir <ATT_CACHE_DIR> [--max_examples N for a quick trial run]
# The same ATT_CACHE_DIR built for the ISAC run can be reused here -- caching is
# independent of num_latent (ISAC.md Sec 3.1).
#
# --use_prj/--prj_dim/--prj_dropout/--distill_loss_div_std are dropped vs. the ISAC
# script: they only affect the implicit-path latent embedding / self-distillation,
# both dead code when num_latent == 0.
#
# Evaluate with test_molsaki.py, not test.py -- this checkpoint was never trained on
# CODI's bot_id/latent/eot_id structure, so it must be evaluated by generating directly
# from the question (see test_molsaki.py's module docstring).

SAVE_DIR=~/codi_ckpt/codi_nl_gpt2_molsaki

# Fill in with the teacher used to build ATT_CACHE_DIR (ISAC.md Sec 3.1/8 -- provenance
# only, never loaded by train.py/CODI -- ISAC.md Sec 7.1 uses the precomputed cache).
TEACHER_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
# Output directory of cache_teacher_attention.py for TEACHER_MODEL_NAME_OR_PATH above.
ATT_CACHE_DIR=~/att_cache/qwen25_7b

mkdir -p "$SAVE_DIR"

cp scripts/train_gpt2_gsm8k-aug-nl_molsaki.sh "$SAVE_DIR"

python train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name gsm8k_gpt2_molsaki_only \
	--logging_dir "$SAVE_DIR/logs"\
	--logging_steps 10 \
	--model_name_or_path gpt2 \
	--data_name icot-full \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 16 \
  	--gradient_accumulation_steps 8 \
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
    --num_latent 0 \
    --logging_strategy "steps" \
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
