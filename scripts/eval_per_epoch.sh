# 2026-10-01: evaluates every epoch-boundary checkpoint produced by one of the *_10ep.sh
# training scripts (--save_strategy epoch, so one checkpoint-N per epoch, 10 total) and
# writes an epoch -> accuracy CSV, so accuracy-vs-epoch trends are visible instead of only
# the final epoch's number.
#
# checkpoint-N directories are named by global *step*, not epoch number -- this script
# sorts them numerically and assigns epoch = rank (1st-smallest-step checkpoint = epoch 1,
# 2nd = epoch 2, ...), which is correct as long as training used --save_strategy epoch
# (exactly one checkpoint per epoch, in order) and ran to completion.
#
# Usage:
#   bash scripts/eval_per_epoch.sh <codi|isac|molsaki> <GPU_ID>
# Examples:
#   bash scripts/eval_per_epoch.sh codi 0
#   bash scripts/eval_per_epoch.sh molsaki 1
#   bash scripts/eval_per_epoch.sh isac 2
# Run the three in parallel (different GPUs) in three separate terminals/tmux panes.
#
# Output: /workspace/per_epoch_eval/<run>_epoch<N>.log, plus a summary CSV at
# /workspace/per_epoch_eval/<run>_accuracy_by_epoch.csv

set -e
RUN=$1
GPU=$2
if [ -z "$RUN" ] || [ -z "$GPU" ]; then
    echo "Usage: bash scripts/eval_per_epoch.sh <codi|isac|molsaki> <GPU_ID>" >&2
    exit 1
fi

cd /workspace/ISAC_implementation
OUT_DIR=/workspace/per_epoch_eval
mkdir -p "$OUT_DIR"

LLAMA1B_COMMON="--data_name gsm8k --output_dir /tmp/test_out --model_name_or_path meta-llama/Llama-3.2-1B-Instruct --seed 11 --model_max_length 512 --bf16 --lora_r 128 --lora_alpha 32 --lora_init --batch_size 128 --greedy True --inf_num_iterations 1 --use_lora True"
LATENT_ARGS="--num_latent 6 --use_prj True --prj_dim 2048 --prj_no_ln False --prj_dropout 0.0 --inf_latent_iterations 6 --remove_eos True"

case "$RUN" in
    codi)
        CKPT_BASE=~/codi_ckpt/codi_nl_llama_10ep/gsm8k_llama1b_latent_baseline/Llama-3.2-1B-Instruct/ep_10/lr_0.0008/seed_11
        EVAL_SCRIPT="test.py"
        EVAL_EXTRA="$LATENT_ARGS"
        ;;
    isac)
        CKPT_BASE=~/codi_ckpt/codi_nl_llama_isac_10ep/gsm8k_llama1b_isac/Llama-3.2-1B-Instruct/ep_10/lr_0.0008/seed_11
        EVAL_SCRIPT="test.py"
        EVAL_EXTRA="$LATENT_ARGS"
        ;;
    molsaki)
        CKPT_BASE=~/codi_ckpt/codi_nl_llama_molsaki_10ep/gsm8k_llama1b_molsaki_only/Llama-3.2-1B-Instruct/ep_10/lr_0.0008/seed_11
        EVAL_SCRIPT="test_molsaki.py"
        EVAL_EXTRA="--num_latent 0 --max_new_tokens 400"
        ;;
    *)
        echo "Unknown run type: $RUN (expected codi|isac|molsaki)" >&2
        exit 1
        ;;
esac

if [ ! -d "$CKPT_BASE" ]; then
    echo "ERROR: $CKPT_BASE does not exist -- has the *_10ep.sh training finished (or even started)?" >&2
    exit 1
fi

# Sort checkpoint-N dirs numerically by N (the step number), not lexicographically
# (checkpoint-2983 must sort before checkpoint-29830).
CKPTS=$(ls -d "$CKPT_BASE"/checkpoint-* 2>/dev/null | sed -E 's#.*/checkpoint-([0-9]+)#\1 &#' | sort -n | awk '{print $2}')

if [ -z "$CKPTS" ]; then
    echo "ERROR: no checkpoint-* directories found under $CKPT_BASE" >&2
    exit 1
fi

SUMMARY="$OUT_DIR/${RUN}_accuracy_by_epoch.csv"
echo "epoch,checkpoint,accuracy" > "$SUMMARY"

EPOCH=1
for CKPT in $CKPTS; do
    LOG="$OUT_DIR/${RUN}_epoch${EPOCH}.log"
    echo "=== $RUN epoch $EPOCH ($CKPT) ==="
    CUDA_VISIBLE_DEVICES=$GPU python "$EVAL_SCRIPT" $LLAMA1B_COMMON $EVAL_EXTRA \
        --ckpt_dir "$CKPT/" \
        > "$LOG" 2>&1
    ACC=$(grep -oE "accuracy: [0-9.]+%" "$LOG" | tail -1 | grep -oE "[0-9.]+")
    echo "$EPOCH,$CKPT,$ACC" >> "$SUMMARY"
    echo "  -> accuracy: ${ACC}%"
    EPOCH=$((EPOCH + 1))
done

echo "Done. Summary: $SUMMARY"
cat "$SUMMARY"
