"""Reasoning-step and critical-token index extraction (ISAC.md Sec 2.1, 3.2, 3.3).

This is a standalone, tokenizer-agnostic utility: it operates purely on
(text, tokenizer) pairs and returns token-index groups. It is deliberately
independent of the teacher-model choice and of `CODI`/`train.py` so it can be
applied identically to the teacher LLM's tokenizer and to the student's
(`self.codi`) tokenizer.

Per ISAC.md Sec 3.3, applying the *same* text-level rule (period splitting for
steps, regex matching for critical tokens) independently to the teacher's and
student's own tokenization of the same underlying text guarantees
`|M1|=|N1|` (same step count) and `|M2|=|N2|` (same critical-token count) by
construction -- so no teacher/student tokenizer alignment is required.

Only `critical_token_mode="numeric"` (ISAC.md Sec 3.2, GSM8K-Aug-NL / math
reasoning branch) is implemented. `"keyword"` mode (CommonsenseQA/StrategyQA)
requires teacher-LLM few-shot keyword-extraction preprocessing and a chosen
teacher model (ISAC.md Sec 3.2, Sec 8 -- out of scope until the teacher model
is decided), so it is left as an explicit NotImplementedError stub.
"""
import re
from typing import List, Sequence, Tuple

import transformers

NUMERIC_TOKEN_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def split_steps_with_offsets(question: str, rationale: str) -> List[Tuple[int, int]]:
    """Segment `question + rationale` into reasoning steps and return each
    step's char span `(start, end)` in the concatenated text.

    Step 0 is always the question (ISAC.md Sec 3.2: "the question itself is
    also counted as a step"). The rationale is then split on periods
    (`". "`), mirroring `train.py`'s existing split for `icot-full`
    (GSM8k-Aug-NL) data (Sec 3.2/Sec 4: "reuse as-is").
    """
    spans = [(0, len(question))]
    offset = len(question)

    parts = rationale.split(". ")
    for idx, part in enumerate(parts):
        if not part.strip():
            offset += len(part) + (len(". ") if idx != len(parts) - 1 else 0)
            continue
        start = offset
        end = start + len(part)
        spans.append((start, end))
        offset = end
        if idx != len(parts) - 1:
            offset += len(". ")

    return spans


def find_numeric_critical_spans(text: str) -> List[Tuple[int, int]]:
    """Find critical-token char spans for math reasoning data (ISAC.md
    Sec 3.2, "Math reasoning (GSM8K-Aug-NL)" branch): every numeric literal
    is a critical token.
    """
    return [m.span() for m in NUMERIC_TOKEN_PATTERN.finditer(text)]


def char_spans_to_token_indices(
    spans: Sequence[Tuple[int, int]],
    offset_mapping: Sequence[Tuple[int, int]],
) -> List[List[int]]:
    """Map char spans to the token indices whose own `(start, end)` offset
    overlaps that span, using a fast tokenizer's `offset_mapping` (ISAC.md
    Sec 3.2: "the tokenizer's char-to-token mapping").

    Special tokens (offset `(0, 0)`) never overlap a real span and are
    naturally excluded.
    """
    groups = []
    for span_start, span_end in spans:
        token_indices = [
            tok_idx
            for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping)
            if tok_end > tok_start and tok_start < span_end and tok_end > span_start
        ]
        groups.append(token_indices)
    return groups


def get_step_and_critical_token_indices(
    question: str,
    rationale: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    critical_token_mode: str = "numeric",
) -> Tuple[List[List[int]], List[List[int]]]:
    """Compute `M1`/`N1` (per-step token-index sets, `K_alpha`) and
    `M2`/`N2` (per-critical-token token-index sets, `P_beta`) for one
    example, for either the teacher or the student tokenizer (ISAC.md
    Sec 2.1, 3.2, 3.3).

    Returns:
        (step_token_indices, critical_token_indices) -- both are lists of
        token-index lists, i.e. `M1`/`N1` and `M2`/`N2` respectively.
    """
    if critical_token_mode != "numeric":
        raise NotImplementedError(
            f"critical_token_mode={critical_token_mode!r} is not implemented. "
            "Only 'numeric' (ISAC.md Sec 3.2, math reasoning branch) is "
            "supported. 'keyword' mode (commonsense/StrategyQA) requires "
            "teacher-LLM few-shot keyword-extraction preprocessing and a "
            "chosen teacher model (ISAC.md Sec 3.2, Sec 8) and is deferred."
        )

    if not tokenizer.is_fast:
        raise ValueError(
            "get_step_and_critical_token_indices requires a fast tokenizer "
            "(needed for return_offsets_mapping, ISAC.md Sec 3.2). Instantiate "
            "the teacher/student tokenizer separately with use_fast=True for "
            "this purpose; this does not require changing the slow tokenizer "
            "used elsewhere in train.py/test.py."
        )

    text = question + rationale
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding["offset_mapping"]

    step_spans = split_steps_with_offsets(question, rationale)
    critical_spans = find_numeric_critical_spans(text)

    step_token_indices = char_spans_to_token_indices(step_spans, offset_mapping)
    critical_token_indices = char_spans_to_token_indices(critical_spans, offset_mapping)

    return step_token_indices, critical_token_indices
