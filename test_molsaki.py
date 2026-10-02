#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Evaluation for the MoLSAKI-only baseline (ISAC.md Sec 5/8, decided 2026-07-14):
# a CODI checkpoint trained with --num_latent 0 (no implicit/latent path) generates
# the explicit natural-language rationale directly, unlike test.py's implicit-path
# eval which alternates bot_id -> latent iterations -> eot_id. This script tokenizes
# the question and autoregressively generates straight through -- no special tokens,
# no latent bookkeeping -- reusing test.py's sampling loop, extract_answer_number,
# and compute_accuracy verbatim (this repo's existing convention already duplicates
# these into probe_latent_token.py rather than sharing a module).

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

do_print = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def evaluation(model_args, data_args, training_args):
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError

    model = CODI(model_args, training_args, lora_config)
    if model_args.ckpt_dir:
        try:
            state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
        except Exception:
            state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))

        # 2026-10-01: see test.py for the full rationale -- strip a DDP-era "module." prefix
        # before loading, and log missing/unexpected keys instead of letting strict=False
        # swallow a full key-name mismatch silently.
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            print(f"[ckpt load] stripped 'module.' prefix from all {len(state_dict)} state_dict keys.")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        lora_missing = [k for k in missing if "lora_" in k]
        print(
            f"[ckpt load] {model_args.ckpt_dir}: {len(state_dict)} keys in file, "
            f"{len(missing)} missing, {len(unexpected)} unexpected."
        )
        if lora_missing:
            print(
                f"[ckpt load] WARNING: {len(lora_missing)} LoRA keys were NOT loaded from the "
                "checkpoint (still at random/zero initialization) -- the model is effectively "
                f"untrained despite --ckpt_dir being set. Sample: {lora_missing[:5]}"
            )
        if unexpected:
            print(f"[ckpt load] Sample unexpected keys in checkpoint (not found in model): {unexpected[:5]}")
    else:
        # 2026-09-30: --ckpt_dir omitted -> "raw" baseline (no CODI/ISAC training at all).
        # init_lora_weights=True zero-inits the LoRA B matrix, so an untrained LoRA adapter
        # contributes nothing and this is equivalent to evaluating the bare base model.
        print("No --ckpt_dir given -- evaluating the untrained base model (raw baseline).")
    model.codi.tie_weights()

    tokenizer_path = model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:  # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"
    if "gsm-hard" == data_args.data_name:
        dataset = load_dataset("juyoung-trl/gsm-hard")
        test_set = dataset['train']
        question_name = "instruction"
        answer_name = "response"
    elif "multi-arith" == data_args.data_name:
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset['test']
        answer_name = "final_ans"
    elif "svamp" == data_args.data_name:
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif "commonsense" == data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset['validation']
    elif "gsm8k" == data_args.data_name:
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset['test']
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    question_raw = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
    answer_raw_examples = list(test_set)

    # 2026-10-01: shard the test set across multiple GPU processes (test_args.num_shards > 1).
    # Interleaved (index % num_shards == shard_id), same convention as
    # cache_teacher_attention.py's sharding, so this is safe even if the underlying dataset has
    # any local ordering structure. No-op when num_shards=1 (default).
    if data_args.num_shards > 1:
        shard_indices = [i for i in range(len(question_raw)) if i % data_args.num_shards == data_args.shard_id]
        question_raw = [question_raw[i] for i in shard_indices]
        answer_raw_examples = [answer_raw_examples[i] for i in shard_indices]
        logging.warning(
            f"[shard {data_args.shard_id}/{data_args.num_shards}] "
            f"evaluating {len(question_raw)} examples (of {len(test_set)} total)"
        )

    # 2026-10-01: raw baseline of an instruct-tuned model (no --ckpt_dir) needs the model's own
    # chat template -- without it, the model treats the bare question as free text to continue
    # rather than as something to answer, which is why the Qwen2.5-7B-Instruct raw-baseline log
    # showed rambling/incomplete generations that ran out `max_new_tokens` instead of emitting
    # EOS. A CODI/MoLSAKI-only checkpoint (--ckpt_dir given) was trained on the bare
    # question+CoT+answer format (train.py's preprocess(), no chat template involved), so the
    # template must NOT be applied there -- it would mismatch what the model was fine-tuned on.
    use_chat_template = (not model_args.ckpt_dir) and getattr(tokenizer, "chat_template", None)
    if use_chat_template:
        logging.warning("No --ckpt_dir + tokenizer has a chat_template -- wrapping questions as chat turns.")
        # 2026-10-01: instruct the model to restate its final answer in the exact
        # "The answer is: <number>" format -- this is the same phrase train.py's own data
        # already uses (get_answer_token_position) and extract_answer_number just grabs the
        # LAST number in the generated text, so without this a model that mentions another
        # number after its answer (e.g. "...5 dozens... for 30 days") gets marked wrong even
        # when its reasoning was correct.
        chat_system_prompt = (
            "Solve the math problem step by step. On the final line, restate the final "
            "numeric answer by itself in the exact format: The answer is: <number>"
        )
        question = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": chat_system_prompt},
                    {"role": "user", "content": q},
                ],
                tokenize=False, add_generation_prompt=True
            )
            for q in question_raw
        ]
    else:
        question = question_raw
    answer = []

    # get numerical answer
    for example in answer_raw_examples:
        example = example[answer_name]
        if isinstance(example, bool):
            answer.append(example)
            continue
        if example in ["True", "False"]:
            if example == "True":
                ans = True
            else:
                ans = False
            answer.append(ans)
            continue
        if example in "ABCDE":
            answer.append(example)
            continue
        if "####" in example:
            ans = example.split('####')[-1]
        else:
            ans = example
        ans = ans.replace(',', '')  # handle numbers like 2,000
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")

    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )

        # MoLSAKI-only: no bot_id append -- the model was never trained on the
        # encoder/latent/decoder split (num_latent == 0), so generation continues
        # directly from the tokenized question.
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to(device))

    model.eval()
    gen_kwargs = {
        "max_new_tokens": training_args.max_new_tokens,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    len_cot = []
    model.eval()

    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        with torch.no_grad():
            # Seed past_key_values by encoding the question directly (no bot_id).
            past_key_values = None
            outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=False, past_key_values=past_key_values, attention_mask=batch["attention_mask"])
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :model.codi.config.vocab_size - 1]

            output = None  # first iteration reuses next_token_logits from the seeding forward above

            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")  # Track EOS for each sequence
            pred_tokens = [[] for _ in range(batch_size)]
            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1

                if output is not None:
                    out = model.codi(
                            inputs_embeds=output,
                            output_hidden_states=False,
                            attention_mask=None,
                            use_cache=True,
                            output_attentions=False,
                            past_key_values=past_key_values
                        )
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :model.codi.config.vocab_size-1]
                else:
                    logits = next_token_logits

                # implement the sampling process
                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Handle EOS for each sequence
                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                # Break if all sequences have finished
                if finished.all():
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            for mini_step, pred_token in enumerate(pred_tokens):
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                if do_print:
                    print(f"Question {step*data_args.batch_size+mini_step} Starts...")
                    print(f"Q: {question_raw[step*data_args.batch_size+mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step*data_args.batch_size+mini_step} Ends")
                    print(f"Prediction={extract_answer_number(decoded_pred)}; Groundtruth={answer[step*data_args.batch_size+mini_step]}")
                    print("")
                ans_pred_list.append(extract_answer_number(decoded_pred))

    accuracy = compute_accuracy(answer, ans_pred_list)
    num_correct = round(accuracy * len(answer))

    # 2026-10-01: print raw correct/total counts (not just %) so parallel shards can be summed
    # correctly afterward (each shard has a different denominator -- averaging percentages
    # directly would weight shards unevenly if sharding is ever uneven).
    print(
        f"[shard {data_args.shard_id}/{data_args.num_shards}] "
        f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*accuracy:.2f}% "
        f"| correct={num_correct} total={len(answer)}"
    )
    print(f"average length of generated text: {sum(len_cot)/len(len_cot)}")

    return 100*accuracy

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        if "commonsense" in data_args.data_name:
            pred = sentence.split("The answer is:")[-1].strip()
            if pred[0] not in "ABCDE":
                return "C"
            return pred[0]
        elif "strategy" in data_args.data_name or "prontoqa" in data_args.data_name.lower():
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
            else:
                raise ValueError
        return float('inf')

    # use the last number as the answer
    pred_answer = float(pred[-1])

    return pred_answer


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1

    return acc / len(gold)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accu_list = []
    for i in range(training_args.inf_num_iterations):
        accu = evaluation(model_args, data_args, training_args)
        accu_list.append(accu)
    print(f"Average accuracy over {training_args.inf_num_iterations} sampling: {sum(accu_list)/len(accu_list)}")
