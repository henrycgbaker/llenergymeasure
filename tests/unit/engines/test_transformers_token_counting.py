"""Tests for PyTorch engine token counting with padded batches (C1 fix)."""

import pytest

torch = pytest.importorskip("torch")


def _count_tokens(inputs: dict, outputs: torch.Tensor) -> tuple[int, int]:
    """Replicate the token counting logic from PyTorchEngine._run_batch()."""
    input_token_count = int(inputs["attention_mask"].sum().item())
    input_lengths = inputs["attention_mask"].sum(dim=1)  # shape: (batch,)
    output_token_count = int(
        sum(max(0, outputs.shape[1] - int(inp_len.item())) for inp_len in input_lengths)
    )
    return input_token_count, output_token_count


def _old_count_tokens(inputs: dict, outputs: torch.Tensor) -> tuple[int, int]:
    """Previous (buggy) logic for comparison."""
    input_token_count = inputs["input_ids"].shape[1] * len(inputs["input_ids"])
    tokens_per_seq = outputs.shape[1] - inputs["input_ids"].shape[1]
    output_token_count = max(0, tokens_per_seq) * len(inputs["input_ids"])
    return input_token_count, output_token_count


class TestUniformBatch:
    """Uniform-length sequences: both methods should agree."""

    def test_input_tokens_uniform(self) -> None:
        # 2 sequences, length 4, all real tokens (no padding)
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = torch.zeros((2, 6), dtype=torch.long)  # 2 new tokens each

        new_count, _ = _count_tokens(inputs, outputs)
        old_count, _ = _old_count_tokens(inputs, outputs)

        assert new_count == 8
        assert new_count == old_count, "Uniform batch: new and old methods must agree"

    def test_output_tokens_uniform(self) -> None:
        # 2 sequences, input length 4, output length 6 → 2 new tokens each
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = torch.zeros((2, 6), dtype=torch.long)

        _, new_out = _count_tokens(inputs, outputs)
        _, old_out = _old_count_tokens(inputs, outputs)

        assert new_out == 4
        assert new_out == old_out, "Uniform batch: output counts must agree"


class TestPaddedBatch:
    """Padded batch: sequences of different lengths padded to the same width."""

    def test_input_tokens_padded_returns_real_tokens_only(self) -> None:
        # Sequence 1: length 3 tokens, padded to 5  → [1, 2, 3, 0, 0], mask [1,1,1,0,0]
        # Sequence 2: length 5 tokens, no padding   → [4, 5, 6, 7, 8], mask [1,1,1,1,1]
        # Total real input tokens = 3 + 5 = 8 (NOT 5*2 = 10)
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = torch.zeros((2, 7), dtype=torch.long)  # dummy outputs

        new_count, _ = _count_tokens(inputs, outputs)
        old_count, _ = _old_count_tokens(inputs, outputs)

        assert new_count == 8, f"Expected 8 real input tokens, got {new_count}"
        assert old_count == 10, "Old method overcounts padded input tokens (expected 10)"
        assert new_count != old_count, "Fix should differ from old method on padded batch"

    def test_output_tokens_padded_per_sequence(self) -> None:
        # Sequence 1: input length 3, padded to 5; total output length 7
        #   → generated tokens for seq1 = 7 - 3 = 4
        # Sequence 2: input length 5 (no padding); total output length 7
        #   → generated tokens for seq2 = 7 - 5 = 2
        # Total output tokens = 4 + 2 = 6 (NOT (7-5)*2 = 4)
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # output tensor shape (batch=2, total_seq_len=7)
        outputs = torch.zeros((2, 7), dtype=torch.long)

        _, new_out = _count_tokens(inputs, outputs)
        _, old_out = _old_count_tokens(inputs, outputs)

        assert new_out == 6, f"Expected 6 output tokens (per-sequence), got {new_out}"
        assert old_out == 4, "Old method undercounts output tokens on padded batch (expected 4)"
        assert new_out != old_out, "Fix should differ from old method on padded output"

    def test_output_tokens_no_new_tokens_clamped_to_zero(self) -> None:
        # Edge case: for a uniform batch where output length equals input length → no new tokens
        # Both sequences are length 4 (no padding), output also length 4
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)  # all real tokens
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # Output same width as input → no generation happened
        outputs = input_ids  # shape (2, 4)

        _, new_out = _count_tokens(inputs, outputs)

        assert new_out == 0, f"Expected 0 output tokens when no generation, got {new_out}"
