"""Teacher LLM attention extraction & caching (ISAC.md Sec 3.1, 7.1).

Forwards each (question, rationale) example from the icot-full
(GSM8K-Aug-NL) dataset through a frozen teacher LLM with
output_attentions=True, aggregates each layer's head-averaged attention
into a step x critical-token matrix `A_l^L` (ISAC.md Sec 2.1, Eq 2/3),
and caches the raw per-layer stack to disk.

This only produces the *raw* per-layer `A_l^L` stack -- MoL layer
combination (Sec 2.2, tau1) is deliberately deferred to training time
(Sec 6 roadmap step 5), since it is a parameter-free statistic computed
from `A_l^L`, so recomputing it doesn't require redoing this (expensive)
teacher forward pass.

Per-example cache files are keyed by the example's *raw* index in the
underlying HF dataset (before any filtering), so a training-time loader
can look up `cache[i]` for the same enumeration order used here,
independent of whatever separate filtering the student pipeline
(`train.py`'s `SupervisedDataset`) applies with its own tokenizer.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm

from src.critical_tokens import get_step_and_critical_token_indices
from src.attention_loss import aggregate_layer_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CacheArguments:
    teacher_model_name_or_path: str = field(
        metadata={"help": "HF model id or local path of the teacher LLM used for attention extraction."}
    )
    data_name: str = field(
        default="icot-full",
        metadata={"help": "Only 'icot-full' (GSM8K-Aug-NL) is supported (ISAC.md Sec 3.2 first-phase scope)."},
    )
    att_cache_dir: str = field(
        default="./att_cache",
        metadata={"help": "Directory to write per-example cached attention tensors to."},
    )
    critical_token_mode: str = field(default="numeric")
    include_last_cot: bool = field(
        default=False, metadata={"help": "Mirror train.py's flag of the same name."}
    )
    max_token_num: int = field(
        default=1000,
        metadata={"help": "Skip examples whose teacher-tokenized (question+rationale) length exceeds this."},
    )
    max_examples: Optional[int] = field(
        default=None, metadata={"help": "Stop after this many raw examples (for smoke-testing)."}
    )
    shard_id: int = field(
        default=0,
        metadata={"help": (
            "2026-09-28: this process's shard index, for running N copies of this script in "
            "parallel across N GPUs (one copy per GPU, via CUDA_VISIBLE_DEVICES). Only examples "
            "with raw_index % num_shards == shard_id are processed by this copy. Sharding is "
            "interleaved (modulo), not contiguous-block, because GSM8K-Aug/-NL augments each base "
            "question into ~50 contiguous near-duplicates -- a contiguous block split would give "
            "different shards very different example distributions (same reasoning as train.py's "
            "exp_mode stride-sampling fix)."
        )},
    )
    num_shards: int = field(
        default=1,
        metadata={"help": "Total number of parallel shards (see shard_id). 1 = original single-process behavior."},
    )


def build_question_and_rationale(example, include_last_cot: bool):
    """Mirror train.py's icot-full branch. 2026-10-01: the final "The answer
    is: X" phrase is now appended as its own trailing step (after the CoT
    steps), not excluded -- must stay byte-for-byte identical to train.py's
    construction (see preprocess() there) so the teacher cache's step/
    critical-token counts match the student's live counts (ISAC.md Sec 3.3
    invariant)."""
    if example["answer"] is None:
        return None, None

    cot = f"{example['cot']}".split(". ")
    if not include_last_cot:
        cot = cot[:-1]

    answer = example["answer"].split(" ")[-1]
    if not answer[0].isdigit():
        return None, None

    question = f"{example['question']}"
    answer_text = f"The answer is: {answer}"
    cot_joined = ". ".join(cot) if cot else ""
    rationale = (cot_joined + ". " + answer_text) if cot_joined else answer_text
    return question, rationale


def main():
    parser = transformers.HfArgumentParser((CacheArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    if args.data_name != "icot-full":
        raise NotImplementedError(
            f"data_name={args.data_name!r} is not supported. ISAC.md Sec 3.2 restricts the "
            "first-phase target dataset to 'icot-full' (GSM8K-Aug-NL); step segmentation for "
            "the word-split 'icot' variant and for commonsense data is not yet defined (Sec 7.4)."
        )

    os.makedirs(args.att_cache_dir, exist_ok=True)

    print(f"Loading teacher model {args.teacher_model_name_or_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.teacher_model_name_or_path, use_fast=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name_or_path, attn_implementation="eager"
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]

    num_cached, num_skipped = 0, 0
    for idx, example in enumerate(tqdm(dataset)):
        if args.max_examples is not None and idx >= args.max_examples:
            break

        if args.num_shards > 1 and idx % args.num_shards != args.shard_id:
            continue

        # 2026-09-28: resume support -- skip examples already cached by a prior (possibly
        # interrupted, e.g. by a paused/resumed rented instance) run. Safe because output is
        # deterministic given (teacher, example, critical_token_mode, include_last_cot).
        out_path = os.path.join(args.att_cache_dir, f"{idx}.pt")
        if os.path.exists(out_path):
            num_cached += 1
            continue

        question, rationale = build_question_and_rationale(example, args.include_last_cot)
        if question is None:
            num_skipped += 1
            continue

        text = question + rationale
        token_num = len(tokenizer.encode(text))
        if token_num > args.max_token_num:
            num_skipped += 1
            continue

        step_groups, critical_groups = get_step_and_critical_token_indices(
            question, rationale, tokenizer, args.critical_token_mode
        )
        if len(critical_groups) == 0:
            # No critical tokens found (e.g. no numeric literal) -- nothing to distill against.
            num_skipped += 1
            continue

        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoding["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        layer_matrices = []
        for layer_attn in outputs.attentions:
            head_avg = layer_attn.mean(dim=1).squeeze(0).to(torch.float32).cpu()  # (seq, seq)
            layer_matrices.append(aggregate_layer_attention(head_avg, step_groups, critical_groups))
        stacked = torch.stack(layer_matrices, dim=0)  # (num_layers, |M1|, |M2|)

        # 2026-09-29: write atomically (temp file + os.replace) so a Ctrl+C / kill / preempted
        # instance mid-write can never leave a half-written file at the final `{idx}.pt` path --
        # this bit us when an interrupted single-GPU caching run left a truncated file that the
        # resume-skip check above (which only checks existence, not validity) silently treated as
        # "already cached" forever, and train.py's forward() (src/model.py, ISAC.md Sec 3.5) later
        # crashed on torch.load with "failed locating file data.pkl". os.replace is atomic on the
        # same filesystem (both paths are under att_cache_dir here), so readers only ever see a
        # complete file or no file at all.
        tmp_path = out_path + ".tmp"
        torch.save(
            {
                "raw_index": idx,
                "attn": stacked,
                "num_steps": len(step_groups),
                "num_critical_tokens": len(critical_groups),
            },
            tmp_path,
        )
        os.replace(tmp_path, out_path)
        num_cached += 1

    meta = {
        "teacher_model_name_or_path": args.teacher_model_name_or_path,
        "data_name": args.data_name,
        "critical_token_mode": args.critical_token_mode,
        "include_last_cot": args.include_last_cot,
        "max_token_num": args.max_token_num,
        "num_cached": num_cached,
        "num_skipped": num_skipped,
    }

    if args.num_shards <= 1:
        # Original single-process behavior, unchanged.
        with open(os.path.join(args.att_cache_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Cached {num_cached} examples, skipped {num_skipped}, to {args.att_cache_dir}")
        return

    # 2026-09-28: parallel-shard mode. Each shard writes its own meta file (filenames of the
    # cached .pt tensors themselves already carry the global raw_index, so those never collide
    # across shards). Once every shard's meta file is present, whichever shard finishes last
    # aggregates them into the single meta.json that CODI's _check_att_cache_consistency (and
    # train.py) actually reads -- so no separate manual merge step is needed.
    shard_meta_path = os.path.join(args.att_cache_dir, f"meta_shard{args.shard_id}.json")
    with open(shard_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[shard {args.shard_id}/{args.num_shards}] cached {num_cached}, skipped {num_skipped}")

    shard_meta_paths = [
        os.path.join(args.att_cache_dir, f"meta_shard{i}.json") for i in range(args.num_shards)
    ]
    if all(os.path.exists(p) for p in shard_meta_paths):
        shard_metas = []
        for p in shard_meta_paths:
            with open(p) as f:
                shard_metas.append(json.load(f))
        merged = dict(shard_metas[0])
        merged["num_cached"] = sum(m["num_cached"] for m in shard_metas)
        merged["num_skipped"] = sum(m["num_skipped"] for m in shard_metas)
        with open(os.path.join(args.att_cache_dir, "meta.json"), "w") as f:
            json.dump(merged, f, indent=2)
        print(
            f"All {args.num_shards} shards done -- merged meta.json written "
            f"(num_cached={merged['num_cached']}, num_skipped={merged['num_skipped']})."
        )


if __name__ == "__main__":
    main()
