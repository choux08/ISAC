import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    get_peft_model,
    PeftModel,
    PeftConfig
)
from torch.nn.functional import gelu
import math
from safetensors.torch import load_file
from transformers.modeling_outputs import ModelOutput
import random
import copy

from src.critical_tokens import get_step_and_critical_token_indices
from src.attention_loss import (
    aggregate_layer_attention,
    combine_layer_attention,
    compute_attention_loss,
    compute_teacher_mol_weights,
)
from src.debug_utils import decode_preview

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2026-08-28: under torchrun DDP (multiple ranks training one shared model), every rank
# executes the same __init__/forward code, so unguarded prints get duplicated once per
# GPU. torchrun sets LOCAL_RANK before any HF/argparse code runs, so this is safe to
# compute at import time; on a plain single-GPU `python train.py` run LOCAL_RANK is
# unset and this is just True, so behavior there is unchanged.
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    full_precision: bool = field(default=True, metadata={"help": "whether use int4 for the base model"})
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    ckpt_dir: Optional[str] = field(default=None, metadata={"help": "checkpoint dir for inference."})

@dataclass
class DataArguments:
    data_name: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "batch size during inference"})
    shard_id: int = field(
        default=0,
        metadata={
            "help": (
                "Which shard this process evaluates, for splitting a test set across multiple "
                "GPUs/processes (e.g. test_molsaki.py). Examples with index % num_shards == "
                "shard_id are evaluated by this copy. 0 = default, no effect when num_shards=1. "
                "Mirrors cache_teacher_attention.py's --shard_id/--num_shards convention."
            )
        },
    )
    num_shards: int = field(
        default=1,
        metadata={"help": "Total number of parallel shards (see shard_id). 1 = original single-process behavior."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    per_device_eval_batch_size: int = field(
        default=1,
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    icot_train_path: str = field(default="/users/k24020023/efficient_cot/icae/code/coconut/icot_gsm8k/train.txt", metadata={"help":"The training data path"})
    num_latent: int = field(default=5, metadata={"help": "The number of latent for training or inference."})
    use_lora: bool = field(default=True, metadata={"help": "Use lora or not."})
    greedy: bool = field(default=False, metadata={"help": "Greedy decoding during inference."})
    exp_mode: bool = field(default=False, metadata={"help": "Use partial number of data. for debugging."})
    exp_data_num: int = field(default=10000, metadata={"help": "The number of data used in exp mode"}) 
    use_prj: bool = field(default=False, metadata={"help": "Use a prj module after the llm for latent generation."}) 
    prj_dim: int = field(default=2048, metadata={"help": "The hidden dim of the projection module."})
    prj_dropout: float = field(default=0.0, metadata={"help": "Dropout ratio of the projection module."})
    prj_no_ln: bool = field(default=False, metadata={"help": "Remove the Layer Norm layer for the projection module."})
    distill_loss_div_std: bool = field(default=False, metadata={"help": "Divide the distillation loss by a std for normallisation."})
    distill_loss_type: str = field(default="smooth_l1", metadata={"help": "Specify the distillation loss. Use smoothL1 by default."})
    distill_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    ref_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    ce_loss_factor: float = field(
        default=1.0,
        metadata={
            "help": (
                "Multiplier of L_ce,i (the implicit/student-path answer CE loss, num_latent>0 "
                "only). 2026-10-01: added so all four loss terms (L_ce,e/ref_loss_factor, "
                "L_ce,i/ce_loss_factor, L_KD/distill_loss_factor, L_att/att_loss_factor) have an "
                "independent weight -- previously L_ce,i was implicitly fixed at 1.0 with no way "
                "to tune it separately. Default 1.0 preserves prior behavior exactly."
            )
        },
    )
    inf_latent_iterations: int = field(default=1, metadata={"help": ""})
    inf_num_iterations: int = field(default=5, metadata={"help": "Run multiple times during inference"})
    max_new_tokens: int = field(
        default=256,
        metadata={
            "help": (
                "2026-10-01: max tokens to generate during inference (test.py, test_molsaki.py, "
                "probe_latent_token.py). Was hardcoded to 256 in each script's gen_kwargs -- too "
                "short for verbose/markdown-style generators (e.g. a raw, untrained Qwen2.5-7B "
                "answering GSM8K gets cut off mid-answer, producing a truncated number). Default "
                "kept at 256 so existing eval behavior is unchanged unless overridden."
            )
        },
    )
    remove_eos: bool = field(default=False, metadata={"help": "Do not add <eos> as a delimiter to split QA."})
    print_ref_model_stats: bool = field(default=False, metadata={"help": "Print some stats for the teacher task."})
    include_last_cot: bool = field(default=False, metadata={"help": "Include the last CoT step in the training data."})
    fix_attn_mask: bool = field(default=False, metadata={"help": "Correct a bug about attention mask."})
    log_full: bool = field(default=False, metadata={"help": "Log all losses."})
    print_loss: bool = field(default=True)
    max_token_num: int = field(default=1000, metadata={"help": "Limit the longest data to avoid OOM."})

    # ISAC (ISAC.md Sec 3.4, roadmap step 4): student-side Mixture-of-Layers for L_att.
    use_student_mol: bool = field(default=False, metadata={"help": "Register the trainable StudentMoL module (ISAC.md Sec 2.2/3.4)."})
    mol_tau_student: float = field(default=0.5, metadata={"help": "Student MoL softmax temperature (ISAC.md Sec 2.4, tau2)."})

    # ISAC (ISAC.md Sec 3.5/3.6, roadmap step 6): L_att integration.
    use_att_loss: bool = field(default=False, metadata={"help": "Enable L_att (teacher<->f_e stepwise attention distillation). Requires use_student_mol=True and att_cache_dir."})
    teacher_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Record of which teacher LLM att_cache_dir was built from (ISAC.md Sec 3.1). Not loaded at train time -- ISAC.md Sec 7.1 adopts caching (see cache_teacher_attention.py) as the default, so CODI only reads att_cache_dir."})
    att_cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory of per-example cached teacher attention produced by cache_teacher_attention.py."})
    att_loss_factor: float = field(default=1.0, metadata={"help": "Weight of L_att (ISAC.md Sec 2.4, MoLSAKI's beta)."})
    mol_tau_teacher: float = field(default=0.1, metadata={"help": "Teacher MoL softmax temperature (ISAC.md Sec 2.4, tau1)."})
    critical_token_mode: str = field(default="numeric", metadata={"help": "'numeric' (math) or 'keyword' (commonsense, not yet implemented). Must match the mode used by cache_teacher_attention.py for the same att_cache_dir."})

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class StudentMoL(nn.Module):
    """Trainable Mixture-of-Layers weighting for the student's (`f_e`)
    attention layers (ISAC.md Sec 2.2 Eq 5, Sec 3.4).

    Unlike the teacher's MoL weights -- parameter-free, gradient-statistic
    based, see `compute_teacher_mol_weights` in `src/attention_loss.py`
    (ISAC.md Sec 2.2) -- the student's layer weights are
    learned jointly with the LoRA parameters, since MoLSAKI's ablations
    (Table 2) show adaptive (learned) layer alignment outperforms any
    fixed single-layer mapping.
    """

    def __init__(self, hidden_dim: int, num_layers: int, tau2: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.tau2 = tau2
        self.rmsnorm = nn.RMSNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim * num_layers, num_layers)

    def forward(self, value_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            value_vectors: per-layer value projections `V_l` (Eq 5), shape
                `(num_layers, batch, seq_len, hidden_dim)`.
        Returns:
            `p^S`, shape `(batch, num_layers)`: softmax layer weights.
        """
        normed = self.rmsnorm(value_vectors)  # (num_layers, batch, seq_len, hidden_dim)
        h = normed.sum(dim=2)  # sum over the sequence dim -> (num_layers, batch, hidden_dim)
        h = h.permute(1, 0, 2).reshape(h.size(1), -1)  # (batch, num_layers*hidden_dim)
        logits = self.linear(h)  # (batch, num_layers)
        return torch.softmax(logits / self.tau2, dim=-1)


class CODI(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        model_wrapper_class = AutoModelForCausalLM
        if IS_MAIN_PROCESS:
            print(f"[CODI.init] loading base model {self.model_name} (full_precision={model_args.full_precision})...")
        if model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    resume_download=True,
                )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    resume_download=True,
                    quantization_config=transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_quant_type='nf4',
                    )
                )
        if IS_MAIN_PROCESS:
            print(f"[CODI.init] base model loaded")

        ori_vocab_size = self.codi.config.vocab_size
        self.training = self.model_args.train

        # special tokens to enclose the latent embeddings
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.eot_id = ori_vocab_size + 2

        self.codi.resize_token_embeddings(
            ori_vocab_size + 3
        )  # dummy values for mem tokens
        if IS_MAIN_PROCESS:
            print(f"[CODI.init] resized token embeddings ({ori_vocab_size} -> {ori_vocab_size + 3})")

        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # LoRA
        if training_args.use_lora:
            if IS_MAIN_PROCESS:
                print("[CODI.init] wrapping with LoRA (get_peft_model)...")
            self.codi = get_peft_model(self.codi, lora_config)
            if IS_MAIN_PROCESS:
                print("[CODI.init] LoRA wrap done")

        # Projection Layer
        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))
            if IS_MAIN_PROCESS:
                print("[CODI.init] projection layer (use_prj) set up")

        # ISAC: student-side Mixture-of-Layers for L_att (ISAC.md Sec 2.2, 3.4)
        self.use_student_mol = training_args.use_student_mol
        if self.use_student_mol:
            num_layers = getattr(self.codi.config, "num_hidden_layers", None)
            if num_layers is None:
                num_layers = getattr(self.codi.config, "n_layer", None)
            if num_layers is None:
                raise ValueError(
                    f"Could not resolve the number of transformer layers for {self.model_name} "
                    "(checked config.num_hidden_layers and config.n_layer) -- StudentMoL needs "
                    "this to size its Linear layer."
                )
            value_dim = self.get_value_dim(self.codi.config)
            self.student_mol = StudentMoL(value_dim, num_layers, tau2=training_args.mol_tau_student)
            if IS_MAIN_PROCESS:
                print(f"[CODI.init] StudentMoL registered (num_layers={num_layers}, value_dim={value_dim})")

        # ISAC: L_att (teacher <-> f_e stepwise attention distillation, ISAC.md Sec 3.5/3.6)
        self.use_att_loss = training_args.use_att_loss
        if self.use_att_loss:
            if not self.use_student_mol:
                raise ValueError(
                    "use_att_loss=True requires use_student_mol=True (ISAC.md Sec 2.4: a "
                    "trainable MoL is required, not optional -- pass both flags together)."
                )
            if training_args.att_cache_dir is None:
                raise ValueError(
                    "use_att_loss=True requires att_cache_dir to be set (ISAC.md Sec 7.1: "
                    "teacher attention is precomputed/cached, not extracted live)."
                )
            if IS_MAIN_PROCESS:
                print(f"[CODI.init] checking att_cache_dir={training_args.att_cache_dir!r} consistency (meta.json)...")
            self._check_att_cache_consistency(training_args)
            if IS_MAIN_PROCESS:
                print("[CODI.init] att_cache_dir consistency check passed")
            self.att_cache_dir = training_args.att_cache_dir
            self.att_loss_factor = training_args.att_loss_factor
            self.mol_tau_teacher = training_args.mol_tau_teacher
            self.critical_token_mode = training_args.critical_token_mode
            # A separate *fast* tokenizer purely for offset-mapping-based critical-token
            # extraction (ISAC.md Sec 3.2) -- self.tokenizer above is intentionally slow
            # (use_fast=False) for the existing CODI pipeline and is left untouched.
            self.fast_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self._captured_values = {}
            self._register_value_hooks()
            if IS_MAIN_PROCESS:
                print(f"[CODI.init] value hooks registered on {len(self._value_hook_handles)} layers")

        # Losses
        # Gate on IS_MAIN_PROCESS too: under DDP every rank would otherwise emit the same
        # [forward]/decode_preview traces once per GPU (see IS_MAIN_PROCESS comment above).
        self.print_loss = training_args.print_loss and IS_MAIN_PROCESS
        self.ref_loss_factor = training_args.ref_loss_factor
        self.ce_loss_factor = training_args.ce_loss_factor

        # Cross Entropy Loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
        
        # Distillation Loss
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError

        # general 
        self.fix_attn_mask = training_args.fix_attn_mask

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.pad_token_id

        if self.training:
            self.init()

        if IS_MAIN_PROCESS:
            print("[CODI.init] CODI construction done")

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name:
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name:
                try:
                    return model.get_base_model().transformer.wte
                except Exception: # no lora
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception: # no lora
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError

    def get_layers(self, model, model_name):
        """Resolve the list of transformer blocks across model families,
        mirroring `get_embd`'s branching (ISAC.md Sec 3.4: value-vector
        extraction needs the same per-family access as embeddings)."""
        try:
            if "pythia" in model_name:
                return model.get_base_model().gpt_neox.layers
            elif "gpt2" in model_name:
                try:
                    return model.get_base_model().transformer.h
                except Exception:  # no lora
                    return model.transformer.h
            else:
                try:
                    return model.get_base_model().model.layers
                except Exception:  # no lora
                    return model.model.layers
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.layers
            raise NotImplementedError

    def get_value_dim(self, config):
        """Resolve the width of a single layer's value projection output
        (ISAC.md Sec 2.2/3.4, `V_l`). For GQA models (e.g. Llama-3.2-1B,
        Mistral, Qwen2 -- `num_key_value_heads < num_attention_heads`) this
        is *smaller* than `config.hidden_size`, so it cannot be assumed to
        equal `self.dim`."""
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        num_heads = getattr(config, "num_attention_heads", None)
        if num_kv_heads is not None and num_heads is not None:
            head_dim = getattr(config, "head_dim", None) or (config.hidden_size // num_heads)
            return num_kv_heads * head_dim
        return config.hidden_size

    def _register_value_hooks(self):
        """Register forward hooks that capture each layer's value-projection
        output `V_l` into `self._captured_values[layer_idx]` (ISAC.md Sec
        2.2 Eq 5, Sec 3.4). Supports GPT-2 (fused `c_attn`, split q/k/v) and
        the Llama/Mistral/Qwen family (separate `v_proj`); Pythia/GPTNeoX's
        interleaved fused layout is out of scope (not one of this repo's two
        target student models per CLAUDE.md)."""
        model_name = self.model_name.lower()
        layers = self.get_layers(self.codi, model_name)
        self._value_hook_handles = []

        if "gpt2" in model_name:
            def make_gpt2_hook(layer_idx):
                def hook(module, inputs, output):
                    embed_dim = output.size(-1) // 3
                    self._captured_values[layer_idx] = output[..., 2 * embed_dim:]
                return hook

            for i, layer in enumerate(layers):
                handle = layer.attn.c_attn.register_forward_hook(make_gpt2_hook(i))
                self._value_hook_handles.append(handle)
        elif "pythia" in model_name:
            raise NotImplementedError(
                "Value-vector extraction for pythia/GPTNeoX's interleaved fused QKV layout "
                "is not implemented (out of scope: CLAUDE.md targets GPT-2 or Llama-3.2-1B-Instruct)."
            )
        else:
            def make_vproj_hook(layer_idx):
                def hook(module, inputs, output):
                    self._captured_values[layer_idx] = output
                return hook

            for i, layer in enumerate(layers):
                handle = layer.self_attn.v_proj.register_forward_hook(make_vproj_hook(i))
                self._value_hook_handles.append(handle)

    def _check_att_cache_consistency(self, training_args):
        """Fail fast if att_cache_dir was built with a different critical-token/CoT
        rule than this run uses (ISAC.md Sec 3.6.1). Without this check, a mismatch
        wouldn't necessarily raise later -- if the cached and live-computed step/
        critical-token counts happen to coincide in size, `compute_attention_loss`'s
        shape check (Sec 3.3) would pass while the content is silently misaligned."""
        meta_path = os.path.join(training_args.att_cache_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise ValueError(
                f"att_cache_dir={training_args.att_cache_dir!r} has no meta.json -- expected "
                "output of cache_teacher_attention.py. Was the cache actually built at this path?"
            )
        with open(meta_path) as f:
            cache_meta = json.load(f)

        if cache_meta.get("critical_token_mode") != training_args.critical_token_mode:
            raise ValueError(
                f"att_cache_dir={training_args.att_cache_dir!r} was built with "
                f"critical_token_mode={cache_meta.get('critical_token_mode')!r}, but this run "
                f"uses critical_token_mode={training_args.critical_token_mode!r}. ISAC.md Sec 3.3's "
                "matching index-count guarantee requires the same rule on both sides -- rebuild "
                "the cache or fix critical_token_mode."
            )
        if cache_meta.get("include_last_cot") != training_args.include_last_cot:
            raise ValueError(
                f"att_cache_dir={training_args.att_cache_dir!r} was built with "
                f"include_last_cot={cache_meta.get('include_last_cot')!r}, but this run uses "
                f"include_last_cot={training_args.include_last_cot!r}. The cached rationale text "
                "won't match what train.py feeds the student, so step/critical-token counts can "
                "silently diverge -- rebuild the cache or fix include_last_cot."
            )
        cached_teacher = cache_meta.get("teacher_model_name_or_path")
        if training_args.teacher_model_name_or_path is not None and cached_teacher != training_args.teacher_model_name_or_path:
            print(
                f"[ISAC WARNING] att_cache_dir={training_args.att_cache_dir!r} was built with "
                f"teacher_model_name_or_path={cached_teacher!r}, but TrainingArguments.teacher_model_name_or_path="
                f"{training_args.teacher_model_name_or_path!r}. This is metadata-only (never loaded), so training "
                "will proceed, but double-check you pointed at the right cache."
            )

    def init(self):
        print_trainable_parameters(self)
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")

    def _dprint(self, msg):
        """Debug-only stage marker, gated by --print_loss (default True). Not part of
        ISAC.md's spec -- purely so a hang/crash mid-forward() shows exactly which stage
        (student pass / teacher pass / ISAC L_att pass / latent loop iteration N) it got
        to, instead of a bare traceback with no execution context."""
        if getattr(self, "print_loss", False):
            print(msg)

    def _decode_preview(self, logits, target_ids_2d, tag, step):
        """Thin wrapper around src/debug_utils.decode_preview -- see that module for
        the actual logic. Kept as a method only so call sites don't need to pass
        self.tokenizer/self.print_loss every time."""
        decode_preview(self.tokenizer, logits, target_ids_2d, tag, step, self.print_loss)

    def forward(
        self,
        encoder_input_ids: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        ref_input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_labels: torch.LongTensor = None,
        step: int = None,
        step_ratio: float = None,
        raw_index: Optional[list] = None,
        question_texts: Optional[list] = None,
        rationale_texts: Optional[list] = None,
    ):
        if not self.fix_attn_mask:
            ref_attention_mask = None

        self._dprint(f"[forward] start (step={step}, num_latent={self.num_latent}, use_att_loss={self.use_att_loss})")

        # Encode the question
        past_key_values = None
        if self.num_latent != 0:
            # No implicit/latent path to feed when num_latent == 0 (MoLSAKI-only baseline,
            # ISAC.md Sec 8 decision 2026-07-14) -- skip this forward entirely rather than
            # compute an unused latent_embd.
            self._dprint("[forward] student encoder pass (question -> bot_id)")
            outputs = self.codi(input_ids=encoder_input_ids, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=encoder_attention_mask)
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1) # as the next input
            if self.use_prj:
                latent_embd = self.prj(latent_embd)
            self._dprint("[forward] student encoder pass done")
        else:
            self._dprint("[forward] num_latent=0 (MoLSAKI-only baseline) -- skipping student encoder pass")

        len_pred_loss = 0
        dynamic_mask = None
        if self.fix_attn_mask:
            dynamic_mask = torch.ones((encoder_attention_mask.size(0), self.num_latent), device=ref_labels.device)

        # Iterate over the latent embeddings
        distill_loss_total = 0
        ce_loss_total = 0
        # Defaults for the num_latent == 0 case (MoLSAKI-only baseline): the loop below never
        # runs, so these would otherwise be referenced before assignment in the debug print
        # and the return dict.
        logits = None
        ce_loss = 0
        distill_loss = 0

        self._dprint("[forward] teacher pass, no_grad (distillation targets)")
        with torch.no_grad():
            ref_outputs = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
        self._dprint("[forward] teacher pass, with grad (ref_ce_loss)")
        ref_outputs_with_grad = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
        self._dprint("[forward] teacher passes done")

        # ISAC: L_att -- teacher <-> f_e stepwise attention distillation (ISAC.md Sec 2.1-2.3, 3.5)
        att_loss_total = 0
        # Debug instrumentation: att_loss is silently 0 (no error, no log) whenever every
        # example in the batch gets skipped below (missing cache file or <2 critical tokens).
        # These counters make that visible instead of it looking like "training but att_loss
        # never moves" -- surfaced in the return dict and in CustomTrainer's logging.
        att_loss_num_examples = 0
        att_loss_num_skipped_no_cache = 0
        att_loss_num_skipped_no_critical = 0
        if self.use_att_loss:
            if raw_index is None or question_texts is None or rationale_texts is None:
                raise ValueError(
                    "use_att_loss=True requires 'raw_index', 'question_texts', and 'rationale_texts' "
                    "in the batch (ISAC.md Sec 3.5) -- these must come from the data pipeline (train.py)."
                )

            # A dedicated, unpadded forward pass per example on (question+rationale) alone,
            # mirroring cache_teacher_attention.py exactly (Sec 3), rather than trying to
            # locate this span inside ref_input_ids -- which interleaves EOS tokens between
            # question/cot/answer (train.py's preprocess()) and would make token-offset
            # alignment with self.fast_tokenizer's char-based spans fragile. Gradients still
            # flow into the same shared self.codi LoRA weights either way.
            self._dprint(f"[forward] ISAC L_att pass starting ({len(raw_index)} examples in batch)")
            att_losses = []
            for i, idx in enumerate(raw_index):
                self._dprint(f"[forward]   L_att example {i+1}/{len(raw_index)} (raw_index={idx})")
                cache_path = os.path.join(self.att_cache_dir, f"{idx}.pt")
                if not os.path.exists(cache_path):
                    att_loss_num_skipped_no_cache += 1
                    continue  # example wasn't cached (e.g. skipped for having no critical tokens)

                question, rationale = question_texts[i], rationale_texts[i]
                step_groups, critical_groups = get_step_and_critical_token_indices(
                    question, rationale, self.fast_tokenizer, self.critical_token_mode
                )
                if len(critical_groups) == 0:
                    att_loss_num_skipped_no_critical += 1
                    continue

                teacher_cache = torch.load(cache_path, map_location=ref_input_ids.device)
                teacher_attn_per_layer = teacher_cache["attn"].to(ref_input_ids.device)

                # Debug check: this is the exact invariant ISAC.md Sec 3.3 relies on (teacher and
                # student, tokenized independently, must agree on step count / critical-token
                # count for the *same* underlying text). If train.py's question/rationale
                # construction ever drifts from cache_teacher_attention.py's (e.g. include_last_cot
                # mismatch that _check_att_cache_consistency didn't catch, or a raw_index pointing
                # at the wrong example), this is where it would first show up -- as a shape
                # mismatch several steps downstream of the real cause. Logging raw_index + both
                # shapes here, instead of just letting compute_attention_loss's ValueError bubble
                # up bare, makes that example directly reproducible via debug_tokenization.py.
                if (
                    teacher_cache["num_steps"] != len(step_groups)
                    or teacher_cache["num_critical_tokens"] != len(critical_groups)
                ):
                    if self.print_loss:
                        print(
                            f"[ISAC WARNING] raw_index={idx}: cached teacher step/critical counts "
                            f"({teacher_cache['num_steps']}, {teacher_cache['num_critical_tokens']}) != "
                            f"live student counts ({len(step_groups)}, {len(critical_groups)}) -- skipping. "
                            "Inspect this example with debug_tokenization.py (ISAC.md Sec 3.3 invariant violated)."
                        )
                    att_loss_num_skipped_no_critical += 1
                    continue

                text = question + rationale
                encoding = self.fast_tokenizer(text, return_tensors="pt", add_special_tokens=False)
                student_input_ids = encoding["input_ids"].to(ref_input_ids.device)

                self._captured_values = {}
                student_outputs = self.codi(input_ids=student_input_ids, output_attentions=True)

                student_layer_attn = [
                    layer_attn.mean(dim=1).squeeze(0).float() for layer_attn in student_outputs.attentions
                ]  # num_layers x (seq, seq), float32 (self.codi may run in fp16/bf16)
                student_attn_per_layer = torch.stack(
                    [
                        aggregate_layer_attention(layer_attn, step_groups, critical_groups)
                        for layer_attn in student_layer_attn
                    ],
                    dim=0,
                )  # (num_layers, N1, N2)

                value_vectors = torch.stack(
                    [self._captured_values[l].float() for l in range(len(student_layer_attn))], dim=0
                )  # (num_layers, 1, seq, value_dim), float32 to match StudentMoL's fp32 params
                p_S = self.student_mol(value_vectors)[0]  # (num_layers,)
                A_S = combine_layer_attention(student_attn_per_layer, p_S)

                p_L = compute_teacher_mol_weights(teacher_attn_per_layer, tau1=self.mol_tau_teacher)
                A_L = combine_layer_attention(teacher_attn_per_layer, p_L)

                att_losses.append(compute_attention_loss(A_L, A_S))

            att_loss_num_examples = len(att_losses)
            self._dprint(f"[forward] ISAC L_att pass done ({att_loss_num_examples}/{len(raw_index)} contributed)")
            if att_losses:
                att_loss_total = torch.stack(att_losses).mean() * self.att_loss_factor
            elif self.print_loss:
                # Previously silent: att_loss_total just stayed 0 with no indication why.
                print(
                    f"[ISAC WARNING] att_loss batch of {len(raw_index)} examples produced 0 "
                    f"valid attention pairs (no_cache={att_loss_num_skipped_no_cache}, "
                    f"no_critical_or_mismatch={att_loss_num_skipped_no_critical}) -- att_loss_total=0 this step."
                )

        # Formatting for deprecated exps
        ref_outputs_list = [ref_outputs] 
        ref_input_ids = [ref_input_ids] 

        # Process the position tensor
        # Normalise the position definition 
        if "llama" in self.model_name.lower() or "qwen" in self.model_name.lower(): # there is one more token standing for " " 
            model_answer_position = model_answer_position + 1
            ref_answer_position = ref_answer_position + 1
       
        # For DEBUG: Print the probability of the teacher task to predict the correct answer
        if self.training_args.print_ref_model_stats:
            for i, (ref_inputs, ref_outputs) in enumerate(zip(ref_input_ids, ref_outputs_list)):
                # evalutae the reference model
                if len(ref_outputs_list) > 1:
                    pos = ref_answer_position[i]
                else:
                    pos = ref_answer_position
                ref_probs = torch.nn.functional.softmax(ref_outputs.logits, dim=-1)
                input_positions = (pos-1).unsqueeze(1).unsqueeze(1).expand(-1, -1, ref_probs.size(2))
                ref_probs_at_positions = ref_probs.gather(1, input_positions)
                probe_positions_positions = pos.unsqueeze(1)
                probe_positions = ref_inputs.gather(1, probe_positions_positions).unsqueeze(1)
                ref_probs_of_target = ref_probs_at_positions.gather(2, probe_positions)
                print(f'stage{i}: mean of the prob of the target token: {ref_probs_of_target.mean()}')
        
        # the model answer position is the position of the eot token to predict the first token of the response
        model_answer_position = model_answer_position - 1
        ref_answer_position = ref_answer_position -1
      
        num_latent = self.num_latent
        if self.num_latent != 0:
            for i in range(num_latent):
                self._dprint(f"[forward] latent step {i+1}/{num_latent}")
                # Implicit CoT generation
                outputs = self.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

                # Calculate the distillation loss
                if i == num_latent - 1: # the last latent embedding
                    self._dprint("[forward] decoding final answer (eot_id + answer) + distill_loss/ce_loss")
                    # Decode the final answer in natural language
                    embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)
                  
                    if dynamic_mask is not None: # Prevent attending the paddings
                        decoder_mask = torch.ones((embds.size(0), embds.size(1)), dtype=torch.bool).to(dynamic_mask)
                        dynamic_mask = torch.cat((encoder_attention_mask, dynamic_mask, decoder_mask), dim=1)
                        dynamic_mask = dynamic_mask.bool()
                    # Student task's output
                    outputs = self.codi(inputs_embeds=embds, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=dynamic_mask) 
                    # Teacher task's output
                    ref_outputs = ref_outputs_list[0]
                    
                    distill_loss = 0
                    # Calculate distillation loss between the teacher's logits and the student's logits for every layer
                    for j, (out, ref_out) in enumerate(zip(outputs.hidden_states, ref_outputs.hidden_states)):
                        ref_selected = ref_out.gather(1, ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1)))
                        out_selected = out.gather(1, model_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1)))

                        distill_loss_tmp = self.distill_loss_fct(out_selected, ref_selected.detach())
                        
                        if self.distill_loss_div_std:
                            if self.distill_loss_type == 'l2':
                                distill_loss_tmp /= ref_selected.std()
                            distill_loss_tmp /= ref_selected.std()
                        distill_loss += distill_loss_tmp
                    
                    distill_loss /= len(outputs.hidden_states)
                    
                    if self.print_loss:
                        print(f'latent{i}: distill_loss={distill_loss}')

                    distill_loss_total += distill_loss

                    # Calculate the CE loss for the student task
                    if i == num_latent - 1:
                        logits = outputs.logits
                        effective_logits = logits[:, :-1, :]
                        effective_logits = effective_logits.reshape(-1, logits.size(-1))
                        target_ids = labels[:, 1:].reshape(-1)                        
                        ce_loss = self.loss_fct(effective_logits, target_ids)
                        ce_loss *= self.ce_loss_factor
                        ce_loss_total += ce_loss
                        self._decode_preview(logits, labels, "student answer (implicit path)", step)

        # Calculate the CE loss for the teacher task
        self._dprint("[forward] computing ref_ce_loss (teacher CE)")
        ref_ce_loss = 0
        ref_logits = ref_outputs_with_grad.logits
        effective_ref_logits = ref_logits[:, :-1, :]
        effective_ref_logits = effective_ref_logits.reshape(-1, ref_logits.size(-1))
        ref_target_ids = ref_labels[:, 1:].reshape(-1)
        ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids)
        ref_ce_loss *= self.ref_loss_factor
        self._decode_preview(ref_logits, ref_labels, "teacher (explicit CoT+answer)", step)

        if logits is None:
            # num_latent == 0 (MoLSAKI-only baseline): the implicit path never produced its
            # own logits, so surface the explicit pass's instead (informational only --
            # CustomTrainer.compute_loss never reads this field).
            logits = ref_logits

        # Weigh the distillation loss
        distill_loss *= self.distill_loss_factor
        distill_loss_total *= self.distill_loss_factor

        if self.print_loss:
            print(f'loss={ce_loss+distill_loss}, ce_loss={ce_loss}, distill_loss={distill_loss}, ce_loss_total={ce_loss_total}, distill_loss_total={distill_loss_total}, ref_ce_loss={ref_ce_loss}, att_loss_total={att_loss_total}, att_loss_num_examples={att_loss_num_examples}')

        loss = ce_loss_total + distill_loss_total + ref_ce_loss + att_loss_total
        self._dprint(f"[forward] done (step={step})")

        if ce_loss_total != 0:
            ce_loss_total = ce_loss_total.detach().item()
        if distill_loss_total != 0:
            distill_loss_total = distill_loss_total.detach().item()
        if ref_ce_loss != 0:
            ref_ce_loss = ref_ce_loss.detach().item()
        if att_loss_total != 0:
            att_loss_total = att_loss_total.detach().item()

        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss_total,
            "distill_loss": distill_loss_total,
            "ref_ce_loss": ref_ce_loss,
            "att_loss": att_loss_total,
            # Debug instrumentation (not part of ISAC.md's spec): how many examples in this
            # batch actually contributed to att_loss_total vs. were silently skipped, and why.
            # CustomTrainer.compute_loss (train.py) logs these alongside the other losses so
            # a healthy vs. starved L_att signal is visible in tensorboard, not just inferred
            # from att_loss hovering near 0.
            "att_loss_num_examples": att_loss_num_examples,
            "att_loss_num_skipped_no_cache": att_loss_num_skipped_no_cache,
            "att_loss_num_skipped_no_critical": att_loss_num_skipped_no_critical,
        }

