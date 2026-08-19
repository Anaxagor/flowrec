"""CPU equivalence checks for DB/SubTB scoring and candidate sampling."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.models.tiger import TigerModel, _CatalogActionIndex


def dense_reference(
    catalog: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    rows, sid_len = targets.shape
    valid = torch.zeros_like(logits, dtype=torch.bool)
    for row in range(rows):
        for depth in range(sid_len):
            if not bool(target_mask[row, depth]):
                continue
            if depth == 0:
                matches = torch.ones(catalog.size(0), dtype=torch.bool)
            else:
                matches = catalog[:, :depth].eq(targets[row, :depth].unsqueeze(0)).all(dim=1)
            allowed = catalog[matches, depth].unique()
            if allowed.numel() == 0:
                allowed = targets[row, depth].reshape(1)
            valid[row, depth, allowed] = True

    constrained = logits.masked_fill(~valid, -torch.inf)
    log_probs = torch.log_softmax(constrained, dim=-1)
    result = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return result.masked_fill(~target_mask, 0.0)


torch.manual_seed(7)
catalog = torch.tensor(
    [
        [1, 10, 20, 30],
        [1, 10, 21, 31],
        [1, 11, 22, 32],
        [2, 12, 23, 33],
        [2, 12, 24, 34],
        [2, 13, 25, 35],
    ],
    dtype=torch.long,
)
targets = torch.cat(
    [
        catalog,
        torch.tensor([[1, 99, 98, 97]], dtype=torch.long),
    ],
    dim=0,
)
target_mask = torch.ones_like(targets, dtype=torch.bool)
target_mask[2, 3] = False
vocab_size = 128

fast_logits = torch.randn(
    targets.size(0),
    targets.size(1),
    vocab_size,
    dtype=torch.float64,
    requires_grad=True,
)
reference_logits = fast_logits.detach().clone().requires_grad_(True)
index = _CatalogActionIndex(catalog)
fast = index.token_logprobs(fast_logits, targets, target_mask)
reference = dense_reference(catalog, reference_logits, targets, target_mask)
assert torch.allclose(fast, reference, atol=1e-12, rtol=1e-12), (fast - reference).abs().max()

coefficients = torch.randn_like(fast)
(fast[:-1] * coefficients[:-1]).sum().backward()
(reference[:-1] * coefficients[:-1]).sum().backward()
assert torch.allclose(fast_logits.grad, reference_logits.grad, atol=1e-12, rtol=1e-12), (
    (fast_logits.grad - reference_logits.grad).abs().max()
)
print("catalog action index forward and gradient equivalence: OK")


half_logits = torch.randn(
    targets.size(0),
    targets.size(1),
    vocab_size,
    dtype=torch.float16,
    requires_grad=True,
)
half_fast = index.token_logprobs(half_logits, targets, target_mask)
half_reference = dense_reference(catalog, half_logits.float(), targets, target_mask)
assert half_fast.dtype == torch.float32
assert torch.allclose(half_fast, half_reference, atol=1e-6, rtol=1e-6)
half_fast[:-1].sum().backward()
assert half_logits.grad is not None
assert torch.isfinite(half_logits.grad).all()
print("low-precision normalization and backward: OK")


semantic_mapping = {
    "a": [1, 10],
    "b": [1, 11],
    "c": [2, 12],
    "d": [2, 13],
}
model = TigerModel(
    codebook_size=16,
    sem_id_len=2,
    user_ids_count=4,
    semantic_id_mapping=semantic_mapping,
    embedding_dim=16,
    num_heads=2,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=32,
    dropout=0.0,
    d_kv=8,
).eval()
batch = {
    "user_ids": torch.tensor([0, 1]),
    "history_token_ids": torch.tensor(
        [
            [[1, 10], [2, 12]],
            [[2, 13], [1, 11]],
        ]
    ),
    "history_token_mask": torch.ones(2, 2, 2, dtype=torch.bool),
    "target_token_ids": torch.tensor([[1, 11], [2, 12]]),
    "target_token_mask": torch.ones(2, 2, dtype=torch.bool),
}
candidate_tokens = torch.tensor(
    [
        [[1, 10], [1, 11], [2, 12]],
        [[2, 13], [1, 11], [2, 12]],
    ]
)
candidate_mask = torch.ones_like(candidate_tokens, dtype=torch.bool)
sft_output = model(batch)
direct_scores = model.score_item_tokens_valid_actions(
    batch,
    candidate_token_ids=candidate_tokens,
    candidate_token_mask=candidate_mask,
)
cached_batch = {
    **batch,
    "_tiger_encoder_last_hidden_state": sft_output.extra["encoder_last_hidden_state"],
    "_tiger_encoder_attention_mask": sft_output.extra["encoder_attention_mask"],
}
cached_scores = model.score_item_tokens_valid_actions(
    cached_batch,
    candidate_token_ids=candidate_tokens,
    candidate_token_mask=candidate_mask,
)
assert torch.allclose(direct_scores, cached_scores, atol=1e-6, rtol=1e-6)
encoder_parameter = next(model.model.encoder.parameters())
direct_grad = torch.autograd.grad(direct_scores.sum(), encoder_parameter, retain_graph=True)[0]
cached_grad = torch.autograd.grad(cached_scores.sum(), encoder_parameter)[0]
assert torch.allclose(direct_grad, cached_grad, atol=1e-5, rtol=1e-5), (direct_grad - cached_grad).abs().max()
print("SFT encoder reuse forward and gradient equivalence: OK")


cfg = GFlowConfig.from_dict({"objective": "db", "reward_cache_path": "unused"})
roundtrip = GFlowConfig.from_dict(cfg.to_dict())
assert roundtrip.objective == "db"
print("speedup configuration roundtrip: OK")
print("ALL DB/SUBTB SPEEDUP SMOKE TESTS PASSED")
