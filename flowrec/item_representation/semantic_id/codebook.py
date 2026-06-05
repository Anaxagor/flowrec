from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

CollisionPolicy = Literal["none", "append", "random_append"]


@dataclass(frozen=True)
class CollisionStats:
    num_items: int
    num_unique_before: int
    num_collisions_before: int
    max_collision_group_size: int
    num_unique_after: int
    num_collisions_after: int
    collision_policy: CollisionPolicy

    def to_dict(self) -> dict[str, int | str]:
        return {
            "num_items": self.num_items,
            "num_unique_before": self.num_unique_before,
            "num_collisions_before": self.num_collisions_before,
            "max_collision_group_size": self.max_collision_group_size,
            "num_unique_after": self.num_unique_after,
            "num_collisions_after": self.num_collisions_after,
            "collision_policy": self.collision_policy,
        }


class SemanticIDCollisionResolver:
    """Resolve duplicate semantic IDs.

    TIGER-style semantic IDs can collide because multiple items may quantize to
    the same tuple. The common pragmatic fix is to append a collision token. We
    ``append`` adds a deterministic within-code rank. ``random_append`` matches
    the TIGER reference notebook: each prefix group gets a random permutation of
    suffix tokens, including singleton groups.

    If tokens from different RQ-VAE levels are flattened into one vocabulary,
    pass ``suffix_offset=num_codebooks * codebook_size`` so collision suffix
    tokens do not overlap with ordinary codebook tokens.
    """

    def __init__(
        self,
        policy: CollisionPolicy = "append",
        *,
        suffix_offset: int = 0,
        suffix_size: int | None = None,
        seed: int = 42,
    ) -> None:
        self.policy = policy
        self.suffix_offset = int(suffix_offset)
        self.suffix_size = suffix_size
        self.seed = int(seed)

    def resolve(
        self,
        item_ids: Sequence[str],
        codes: torch.LongTensor,
    ) -> tuple[torch.LongTensor, CollisionStats]:
        if codes.ndim != 2:
            raise ValueError(f"codes must have shape [num_items, sid_len], got {codes.shape}")
        if len(item_ids) != int(codes.shape[0]):
            raise ValueError("item_ids length must match number of code rows")

        tuples = [tuple(int(x) for x in row.tolist()) for row in codes.cpu()]
        counts = Counter(tuples)
        num_unique_before = len(counts)
        num_collisions_before = len(tuples) - num_unique_before
        max_group = max(counts.values()) if counts else 0

        if self.policy == "none":
            resolved = codes.detach().cpu().long()
        elif self.policy == "append":
            group_positions: dict[tuple[int, ...], int] = defaultdict(int)
            appended: list[list[int]] = []
            # Stable order: row order from embedding store, which should be deterministic.
            for code_tuple in tuples:
                rank = group_positions[code_tuple]
                group_positions[code_tuple] += 1
                appended.append(list(code_tuple) + [self.suffix_offset + rank])
            resolved = torch.tensor(appended, dtype=torch.long)
        elif self.policy == "random_append":
            suffix_size = self.suffix_size
            if suffix_size is None:
                suffix_size = max_group
            if max_group > suffix_size:
                raise ValueError(
                    f"Collision group of size {max_group} cannot fit in suffix_size={suffix_size}"
                )

            grouped_rows: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for row_idx, code_tuple in enumerate(tuples):
                grouped_rows[code_tuple].append(row_idx)

            generator = torch.Generator().manual_seed(self.seed)
            appended = [list(code_tuple) + [self.suffix_offset] for code_tuple in tuples]
            for code_tuple, row_indices in grouped_rows.items():
                suffixes = torch.randperm(int(suffix_size), generator=generator)[: len(row_indices)]
                for row_idx, suffix in zip(row_indices, suffixes.tolist()):
                    appended[row_idx] = list(code_tuple) + [self.suffix_offset + int(suffix)]
            resolved = torch.tensor(appended, dtype=torch.long)
        else:
            raise ValueError(f"Unknown collision policy: {self.policy}")

        after_tuples = [tuple(int(x) for x in row.tolist()) for row in resolved]
        after_counts = Counter(after_tuples)
        stats = CollisionStats(
            num_items=len(item_ids),
            num_unique_before=num_unique_before,
            num_collisions_before=num_collisions_before,
            max_collision_group_size=max_group,
            num_unique_after=len(after_counts),
            num_collisions_after=len(after_tuples) - len(after_counts),
            collision_policy=self.policy,
        )
        return resolved, stats


def offset_code_tokens(codes: torch.LongTensor, *, codebook_size: int) -> torch.LongTensor:
    """Convert per-level RQ-VAE codes into a flat token vocabulary.

    For L codebooks of size K, raw codes are in [0, K). This function returns
    code_l + l * K so the same integer from different codebooks becomes a
    distinct token. This matches the common TIGER-style vocabulary design.
    """
    if codes.ndim != 2:
        raise ValueError(f"codes must have shape [num_items, sid_len], got {codes.shape}")
    offsets = torch.arange(codes.shape[1], dtype=torch.long, device=codes.device).view(1, -1)
    return codes.long() + offsets * int(codebook_size)


def codes_to_mapping(item_ids: Sequence[str], codes: torch.LongTensor) -> dict[str, list[int]]:
    if codes.ndim != 2:
        raise ValueError(f"codes must have shape [num_items, sid_len], got {codes.shape}")
    if len(item_ids) != int(codes.shape[0]):
        raise ValueError("item_ids length must match number of code rows")
    return {
        str(item_id): [int(x) for x in codes[row_idx].tolist()]
        for row_idx, item_id in enumerate(item_ids)
    }
