"""Regression checks for src/critical_tokens.py (ISAC.md Sec 3.2, 3.3).

Self-contained: uses a fake char-level tokenizer, no model/network download
needed. Run directly: `python tests/test_critical_tokens.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.critical_tokens import (
    char_spans_to_token_indices,
    find_numeric_critical_spans,
    get_step_and_critical_token_indices,
    split_steps_with_offsets,
)


class FakeCharTokenizer:
    """One 'token' per character -- makes offsets trivial to reason about."""

    is_fast = True

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class FakeSlowTokenizer:
    is_fast = False


def test_step_split_and_critical_spans():
    question = "Natalia sold clips to 48 of her friends in April."
    rationale = "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether.\n"
    text = question + rationale

    steps = split_steps_with_offsets(question, rationale)
    assert len(steps) == 3, f"expected question + 2 rationale steps, got {len(steps)}"
    assert text[steps[0][0] : steps[0][1]] == question

    crits = find_numeric_critical_spans(text)
    expected_numbers = ["48", "48", "2", "24", "48", "24", "72"]
    found_numbers = [text[s:e] for s, e in crits]
    assert found_numbers == expected_numbers, f"expected {expected_numbers}, got {found_numbers}"
    print("OK: step split (3 steps) and numeric critical spans (7 numbers) match expectations")


def test_char_span_to_token_mapping():
    text = "abc 123 def"
    offset_mapping = [(i, i + 1) for i in range(len(text))]
    spans = find_numeric_critical_spans(text)
    groups = char_spans_to_token_indices(spans, offset_mapping)
    assert groups == [[4, 5, 6]], f"expected token indices [4,5,6] for '123', got {groups}"
    print("OK: char-span -> token-index mapping is correct")


def test_get_step_and_critical_token_indices_end_to_end():
    tok = FakeCharTokenizer()
    q = "What is 2 plus 2."
    r = "First we add 2 and 2. Then the sum is 4.\n"
    steps, crits = get_step_and_critical_token_indices(q, r, tok, "numeric")
    assert len(steps) == 3
    assert len(crits) == 5  # 2, 2, 2, 2, 4
    print("OK: get_step_and_critical_token_indices end-to-end (3 steps, 5 critical tokens)")


def test_keyword_mode_not_implemented():
    tok = FakeCharTokenizer()
    try:
        get_step_and_critical_token_indices("q", "r", tok, "keyword")
        raise AssertionError("expected NotImplementedError for critical_token_mode='keyword'")
    except NotImplementedError:
        print("OK: critical_token_mode='keyword' raises NotImplementedError (deferred, ISAC.md Sec 3.2)")


def test_slow_tokenizer_rejected():
    try:
        get_step_and_critical_token_indices("q", "r", FakeSlowTokenizer(), "numeric")
        raise AssertionError("expected ValueError for a slow (non-fast) tokenizer")
    except ValueError:
        print("OK: a slow tokenizer is rejected with a clear ValueError")


if __name__ == "__main__":
    test_step_split_and_critical_spans()
    test_char_span_to_token_mapping()
    test_get_step_and_critical_token_indices_end_to_end()
    test_keyword_mode_not_implemented()
    test_slow_tokenizer_rejected()
    print("ALL CHECKS PASSED")
