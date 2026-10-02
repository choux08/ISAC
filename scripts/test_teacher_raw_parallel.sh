# 2026-10-01: parallel launcher for the teacher's raw-baseline GSM8K test
# (test_molsaki.py with no --ckpt_dir, i.e. evaluating Qwen2.5-7B-Instruct's own ability,
# not a CODI/MoLSAKI checkpoint). Splits the GSM8K test set across N GPUs using
# test_molsaki.py's new --shard_id/--num_shards (src/model.py DataArguments), interleaved
# the same way cache_teacher_attention.py shards (index % num_shards == shard_id).
#
# This is safe to parallelize purely because it's read-only evaluation (no shared
# checkpoint/optimizer state, no writes to a common directory) -- unlike training, there's no
# DDP/gradient-sync reason to keep it on one process.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/test_teacher_raw_parallel.sh
# Adjust NUM_SHARDS (and the GPU list) if you have a different number of GPUs -- it must match
# the number of comma-separated indices in CUDA_VISIBLE_DEVICES exactly.
#
# After all shards finish, sum accuracy correctly (shards may have slightly different sizes if
# the test set size isn't divisible by NUM_SHARDS) with:
#   grep "correct=" test_teacher_raw_shard*.log | awk -F'correct=|total=' '{c+=$2; t+=$3} END{print c"/"t" = "100*c/t"%"}'

MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
NUM_SHARDS=4

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#GPUS[@]}" -ne "$NUM_SHARDS" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES has ${#GPUS[@]} GPU(s) but NUM_SHARDS=$NUM_SHARDS. Set both consistently." >&2
    exit 1
fi

PIDS=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPUS[$shard_id]}"
    echo "Launching shard $shard_id on GPU $gpu -> test_teacher_raw_shard${shard_id}.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python test_molsaki.py \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --lora_init True \
        --data_name gsm8k \
        --batch_size 8 \
        --max_new_tokens 512 \
        --inf_num_iterations 1 \
        --shard_id "$shard_id" \
        --num_shards "$NUM_SHARDS" \
        > "test_teacher_raw_shard${shard_id}.log" 2>&1 &
    PIDS+=($!)
done

echo "All $NUM_SHARDS shards launched (PIDs: ${PIDS[*]}). Waiting for completion..."
wait "${PIDS[@]}"
echo "All shards finished. Combine accuracy with:"
echo '  grep "correct=" test_teacher_raw_shard*.log | awk -F"correct=|total=" '"'"'{c+=$2; t+=$3} END{print c"/"t" = "100*c/t"%"}'"'"
