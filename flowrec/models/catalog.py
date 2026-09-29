"""Shared catalog-prefix normalization for generative recommenders."""

import torch


class _CatalogActionIndex:
    """Compact prefix-to-valid-children index for catalog-constrained scoring."""

    def __init__(self, semantic_ids: torch.Tensor) -> None:
        if semantic_ids.ndim != 2 or semantic_ids.numel() == 0:
            raise ValueError("semantic_ids must have shape [num_items, sid_len] and be non-empty.")
        tokens = semantic_ids.detach().long().cpu()
        if int(tokens.min()) < 0:
            raise ValueError("semantic_ids must be non-negative.")

        self.sid_len = int(tokens.size(1))
        self.max_token = int(tokens.max())
        self.key_bits = max((self.max_token + 1).bit_length(), 1)
        if self.key_bits * self.sid_len > 62:
            raise ValueError(
                "Cannot pack semantic prefixes into int64: " "{} levels x {} bits.".format(self.sid_len, self.key_bits)
            )
        self._token_mask = (1 << self.key_bits) - 1
        self._cpu_levels = self._build_levels(tokens)
        self._device_levels: dict[torch.device, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def _pack(self, tokens: torch.Tensor, depth: int) -> torch.Tensor:
        keys = torch.zeros(tokens.size(0), dtype=torch.int64, device=tokens.device)
        for level in range(depth):
            keys = (keys << self.key_bits) | (tokens[:, level].long() + 1)
        return keys

    def _build_levels(
        self,
        tokens: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        levels: list[tuple[torch.Tensor, torch.Tensor]] = []
        for depth in range(self.sid_len):
            # A packed edge is the parent-prefix key with one child token appended.
            packed_edges = torch.unique(self._pack(tokens, depth + 1), sorted=True)
            prefix_keys = packed_edges >> self.key_bits
            child_tokens = (packed_edges & self._token_mask) - 1
            node_keys, child_counts = torch.unique_consecutive(prefix_keys, return_counts=True)
            max_children = int(child_counts.max())
            group_ids = torch.repeat_interleave(torch.arange(node_keys.numel(), dtype=torch.long), child_counts)
            group_starts = torch.repeat_interleave(child_counts.cumsum(0) - child_counts, child_counts)
            offsets = torch.arange(packed_edges.numel(), dtype=torch.long) - group_starts
            child_table = torch.full((node_keys.numel(), max_children), -1, dtype=torch.long)
            child_table[group_ids, offsets] = child_tokens
            levels.append((node_keys.contiguous(), child_table.contiguous()))
        return levels

    def _levels_on(
        self,
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        levels = self._device_levels.get(device)
        if levels is None:
            levels = [(keys.to(device=device), children.to(device=device)) for keys, children in self._cpu_levels]
            self._device_levels[device] = levels
        return levels

    def token_logprobs(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize target logits over valid catalog children without a dense mask."""
        if logits.ndim != 3:
            raise ValueError("logits must have shape [num_rows, sid_len, vocab_size].")
        if targets.shape != logits.shape[:2] or target_mask.shape != targets.shape:
            raise ValueError("targets and target_mask must match logits[:2].")
        if int(logits.size(1)) > self.sid_len:
            raise ValueError(
                "Target SID length {} exceeds catalog length {}.".format(int(logits.size(1)), self.sid_len)
            )
        if self.max_token >= int(logits.size(-1)):
            raise ValueError(
                "Catalog token {} exceeds logits vocabulary {}.".format(self.max_token, int(logits.size(-1)))
            )

        targets = targets.to(device=logits.device).long()
        target_mask = target_mask.to(device=logits.device).bool()
        levels = self._levels_on(logits.device)
        per_step: list[torch.Tensor] = []
        num_rows = int(logits.size(0))
        normalization_dtype = torch.float32 if logits.dtype in {torch.float16, torch.bfloat16} else logits.dtype

        for depth in range(int(logits.size(1))):
            active_idx = target_mask[:, depth].nonzero(as_tuple=False).flatten()
            if active_idx.numel() == 0:
                per_step.append(torch.zeros(num_rows, device=logits.device, dtype=normalization_dtype))
                continue

            active_targets = targets.index_select(0, active_idx)
            prefix_keys = self._pack(active_targets, depth)
            node_keys, child_table = levels[depth]
            positions = torch.searchsorted(node_keys, prefix_keys)
            in_range = positions.lt(node_keys.numel())
            safe_positions = positions.clamp_max(node_keys.numel() - 1)
            found = in_range & node_keys.index_select(0, safe_positions).eq(prefix_keys)

            children = child_table.index_select(0, safe_positions)
            first_column = torch.arange(children.size(1), device=children.device).eq(0).unsqueeze(0)
            fallback = ~found.unsqueeze(1)
            child_mask = (children.ge(0) & found.unsqueeze(1)) | (fallback & first_column)
            child_ids = torch.where(
                fallback & first_column,
                active_targets[:, depth].unsqueeze(1),
                children,
            ).clamp_min(0)

            active_logits = logits.index_select(0, active_idx)[:, depth, :].to(dtype=normalization_dtype)
            allowed_logits = active_logits.gather(1, child_ids)
            log_denom = torch.logsumexp(allowed_logits.masked_fill(~child_mask, -torch.inf), dim=1)
            target_logits = active_logits.gather(1, active_targets[:, depth].unsqueeze(1)).squeeze(1)
            target_allowed = (child_ids.eq(active_targets[:, depth].unsqueeze(1)) & child_mask).any(dim=1)
            step_values = torch.where(
                target_allowed,
                target_logits - log_denom,
                target_logits.new_full(target_logits.shape, -torch.inf),
            )
            per_step.append(step_values.new_zeros((num_rows,)).index_copy(0, active_idx, step_values))

        return torch.stack(per_step, dim=1)
