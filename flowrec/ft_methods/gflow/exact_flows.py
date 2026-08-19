from __future__ import annotations

import math

import torch


class CatalogPrefixFlowIndex:
    """Exact prefix flows over a fixed SID catalog via lexicographic ranges.

    Catalog SIDs are sorted once by packed integer keys, so every prefix maps
    to a contiguous range of sorted rows. Per-user flows are then range sums
    over a cumulative-sum of that user's dense reward vector:

        F(prefix | user) = sum_{items under prefix} R(item | user)

    All outputs are exact and carry no gradients. Candidate prefixes that do
    not exist in the catalog get flow = reward_epsilon (log-clamped).
    """

    def __init__(self, item_tokens: torch.Tensor) -> None:
        if item_tokens.ndim != 2:
            raise ValueError("item_tokens must have shape [num_items, sid_len].")
        if item_tokens.numel() == 0:
            raise ValueError("item_tokens must be non-empty.")
        tokens = item_tokens.detach().long().cpu()
        if int(tokens.min()) < 0:
            raise ValueError("item_tokens must be non-negative for packed-key sorting.")
        self.num_items, self.sid_len = int(tokens.size(0)), int(tokens.size(1))
        max_token = int(tokens.max())
        self.key_bits = max(int(math.ceil(math.log2(max_token + 2))), 1)
        if self.key_bits * self.sid_len > 62:
            raise ValueError(f"Cannot pack SIDs into int64 keys: {self.sid_len} levels x {self.key_bits} bits.")
        full_keys = self._pack(tokens, self.sid_len)
        self.perm = torch.argsort(full_keys)
        sorted_tokens = tokens.index_select(0, self.perm)
        # sorted_prefix_keys[d] is a non-decreasing [num_items] key array for depth d+1.
        self.sorted_prefix_keys = [
            self._pack(sorted_tokens, depth).contiguous() for depth in range(1, self.sid_len + 1)
        ]
        self._device_cache: dict[torch.device, tuple[torch.Tensor, list[torch.Tensor]]] = {}

    def _pack(self, tokens: torch.Tensor, depth: int) -> torch.Tensor:
        """Pack the first `depth` token levels into a single int64 key per row."""
        keys = torch.zeros(tokens.size(0), dtype=torch.int64, device=tokens.device)
        for level in range(depth):
            keys = (keys << self.key_bits) | (tokens[:, level].long() + 1)
        # Left-align so that shorter prefixes order consistently with full keys.
        keys = keys << (self.key_bits * (self.sid_len - depth))
        return keys

    def _tensors_on(self, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
        cached = self._device_cache.get(device)
        if cached is None:
            cached = (
                self.perm.to(device),
                [keys.to(device) for keys in self.sorted_prefix_keys],
            )
            self._device_cache[device] = cached
        return cached

    @torch.no_grad()
    def prefix_log_flows(
        self,
        candidate_tokens: torch.Tensor,
        dense_rewards: torch.Tensor,
        *,
        reward_epsilon: float,
    ) -> torch.Tensor:
        """Exact log F for all prefix depths. [B, M, L] tokens -> [B, M, L+1] log flows."""
        if candidate_tokens.ndim != 3:
            raise ValueError("candidate_tokens must have shape [B, M, L].")
        if candidate_tokens.size(-1) != self.sid_len:
            raise ValueError(f"candidate SID length {int(candidate_tokens.size(-1))} != catalog {self.sid_len}.")
        if dense_rewards.ndim != 2 or int(dense_rewards.size(1)) != self.num_items:
            raise ValueError("dense_rewards must have shape [B, num_items].")
        device = candidate_tokens.device
        perm, sorted_keys = self._tensors_on(device)
        bsz, num_cands, _ = candidate_tokens.shape

        sorted_rewards = dense_rewards.float().index_select(1, perm)
        csum = torch.cat(
            [
                torch.zeros((bsz, 1), device=device, dtype=torch.float64),
                sorted_rewards.double().cumsum(dim=1),
            ],
            dim=1,
        )

        eps = float(reward_epsilon)
        out = torch.empty((bsz, num_cands, self.sid_len + 1), device=device, dtype=torch.float32)
        out[..., 0] = csum[:, -1].clamp_min(eps).log().float().unsqueeze(1)
        flat_tokens = candidate_tokens.reshape(bsz * num_cands, self.sid_len).long()
        for depth in range(1, self.sid_len + 1):
            keys = self._pack(flat_tokens, depth)
            ranks = sorted_keys[depth - 1]
            lo = torch.searchsorted(ranks, keys, side="left").reshape(bsz, num_cands)
            hi = torch.searchsorted(ranks, keys, side="right").reshape(bsz, num_cands)
            flow = csum.gather(1, hi) - csum.gather(1, lo)
            out[..., depth] = flow.clamp_min(eps).log().float()
        return out
