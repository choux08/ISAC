# 2026-10-01: OOD math benchmark sweep (SVAMP, GSM-Hard, MultiArith) across all five
# checkpoints in the comparison: raw Llama-3.2-1B-Instruct, CODI-only, MoLSAKI-only, ISAC,
# and the raw teacher (Qwen2.5-7B-Instruct). GSM8K itself isn't included here -- already have
# those numbers from earlier runs.
#
# The four Llama-1B-scale evals (raw/codi/molsaki/isac) run in parallel on GPUs 0-3 per
# dataset (small model, fits easily, and this is read-only eval so no DDP/gradient-sync
# concerns -- same reasoning as the teacher raw-baseline sharding earlier). The 7B teacher
# runs by itself afterward (needs more VRAM headroom, and splitting it across 4 GPUs here
# would just add complexity for a comparatively quick eval at batch_size 128... adjust
# TEACHER_BATCH_SIZE down if it OOMs on a single GPU).
#
# Usage: bash scripts/eval_all_ood.sh
# Logs: /workspace/ood_eval_logs/{model}_{dataset}.log

set -e
cd /workspace/ISAC_implementation

CODI_ONLY_CKPT=~/codi_ckpt/codi_nl_llama/gsm8k_llama1b_latent_baseline/Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11/
MOLSAKI_ONLY_CKPT=~/codi_ckpt/codi_nl_llama_molsaki_full/gsm8k_llama1b_molsaki_only/Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11/
ISAC_CKPT=~/codi_ckpt/codi_nl_llama_isac_full/gsm8k_llama1b_isac/Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11/
TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct

LOG_DIR=/workspace/ood_eval_logs
mkdir -p "$LOG_DIR"

LLAMA1B_COMMON="--model_name_or_path meta-llama/Llama-3.2-1B-Instruct --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 128 --greedy True --inf_num_iterations 1 --use_lora True"
LATENT_ARGS="--num_latent 6 --use_prj True --prj_dim 2048 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True"

for DATASET in svamp gsm-hard multi-arith; do
    echo "=== $DATASET ==="

    CUDA_VISIBLE_DEVICES=0 python test_molsaki.py --data_name "$DATASET" --output_dir /tmp/test_out $LLAMA1B_COMMON \
        > "$LOG_DIR/raw_llama_${DATASET}.log" 2>&1 &
    PID0=$!

    CUDA_VISIBLE_DEVICES=1 python test.py --data_name "$DATASET" --output_dir /tmp/test_out $LLAMA1B_COMMON $LATENT_ARGS \
        --ckpt_dir "$CODI_ONLY_CKPT" \
        > "$LOG_DIR/codi_only_${DATASET}.log" 2>&1 &
    PID1=$!

    CUDA_VISIBLE_DEVICES=2 python test_molsaki.py --data_name "$DATASET" --output_dir /tmp/test_out $LLAMA1B_COMMON --num_latent 0 --max_new_tokens 400 \
        --ckpt_dir "$MOLSAKI_ONLY_CKPT" \
        > "$LOG_DIR/molsaki_only_${DATASET}.log" 2>&1 &
    PID2=$!

    CUDA_VISIBLE_DEVICES=3 python test.py --data_name "$DATASET" --output_dir /tmp/test_out $LLAMA1B_COMMON $LATENT_ARGS \
        --ckpt_dir "$ISAC_CKPT" \
        > "$LOG_DIR/isac_${DATASET}.log" 2>&1 &
    PID3=$!

    echo "Launched raw_llama(pid $PID0) codi_only(pid $PID1) molsaki_only(pid $PID2) isac(pid $PID3) on $DATASET -- waiting..."
    wait $PID0 $PID1 $PID2 $PID3
    echo "$DATASET: 1B-scale models done."

    CUDA_VISIBLE_DEVICES=0 python test_molsaki.py --data_name "$DATASET" --output_dir /tmp/test_out \
        --model_name_or_path "$TEACHER_MODEL" --seed 11 --model_max_length 512 --bf16 \
        --lora_r 128 --lora_alpha 32 --lora_init --batch_size 8 --greedy True --inf_num_iterations 1 --use_lora True \
        --max_new_tokens 512 \
        > "$LOG_DIR/teacher_raw_${DATASET}.log" 2>&1
    echo "$DATASET: teacher done."
done

echo "All done. Summary:"
grep -H "accuracy:" "$LOG_DIR"/*.log
