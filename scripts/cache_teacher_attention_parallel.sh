# 2026-09-28: parallel launcher for cache_teacher_attention.py's new --shard_id/--num_shards
# option. Runs one process per GPU, each handling 1/N of the dataset (interleaved by raw
# index, so GSM8K-Aug-NL's ~50x-contiguous augmentation doesn't skew any one shard -- see
# cache_teacher_attention.py's shard_id field docstring). All shards write into the same
# ATT_CACHE_DIR; per-example .pt files are keyed by global raw_index so there's no collision,
# and the last shard to finish automatically merges the per-shard meta files into meta.json.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/cache_teacher_attention_parallel.sh
# Adjust NUM_SHARDS below (and the GPU list) if you have a different number of GPUs -- it
# must match the number of comma-separated indices in CUDA_VISIBLE_DEVICES exactly.
#
# Logs go to $ATT_CACHE_DIR/shard{0..N-1}.log. Check progress with:
#   tail -f ~/att_cache/qwen25_7b/shard*.log
# and confirm completion with:
#   cat ~/att_cache/qwen25_7b/meta.json

TEACHER_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
ATT_CACHE_DIR=~/att_cache/qwen25_7b
NUM_SHARDS=4

mkdir -p "$ATT_CACHE_DIR"

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#GPUS[@]}" -ne "$NUM_SHARDS" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES has ${#GPUS[@]} GPU(s) but NUM_SHARDS=$NUM_SHARDS. Set both consistently." >&2
    exit 1
fi

PIDS=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPUS[$shard_id]}"
    echo "Launching shard $shard_id on GPU $gpu -> $ATT_CACHE_DIR/shard${shard_id}.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python cache_teacher_attention.py \
        --teacher_model_name_or_path "$TEACHER_MODEL_NAME_OR_PATH" \
        --att_cache_dir "$ATT_CACHE_DIR" \
        --shard_id "$shard_id" \
        --num_shards "$NUM_SHARDS" \
        > "$ATT_CACHE_DIR/shard${shard_id}.log" 2>&1 &
    PIDS+=($!)
done

echo "All $NUM_SHARDS shards launched (PIDs: ${PIDS[*]}). Waiting for completion..."
wait "${PIDS[@]}"
echo "All shards finished. Check $ATT_CACHE_DIR/meta.json for the merged summary."
