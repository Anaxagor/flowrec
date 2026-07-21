from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import torch
import torch.nn as nn

from flowrec.models.base import Batch
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


SimilarityMode = Literal["exact", "hierarchical", "embedding_cosine"]


@dataclass(frozen=True, slots=True)
class SimilarityRewardConfig:
    """Terminal similarity-to-target reward for semantic-ID trajectories.

    Three modes (see examples/amazon_reward_analysis.ipynb, sections 5.2-5.5,
    for the analysis behind each design choice):

    ``exact``
        Positionwise token agreement with the target's semantic ID, scaled to
        [0, 1]: ``#matching positions / L``. Near-binary in practice - the
        catalog collapses onto the epsilon floor with a thin graded shoulder.

    ``hierarchical``
        Prefix-gated weighted Gower over the RQ levels. A level contributes
        only while the whole prefix above it matches; the first divergent RQ
        level contributes its codebook-geometry similarity
        ``1 - ||e(c_a) - e(c_b)|| / max||.||``; every level after the first
        divergence contributes 0. Levels beyond the RQ codebooks (e.g. a
        collision-disambiguation suffix) always use exact match. Requires the
        RQ-VAE ``codebooks`` tensor. Best-founded token-only similarity:
        geometry is graded exactly where it was shown to order item
        similarity (the level-1 divergence), and deeper evidence is discrete.

    ``embedding_cosine``
        Similarity in the raw content-embedding space, skipping the quantizer:
        ``(cos(z_a, z_t) + 1) / 2`` on the (normalized) item content
        embeddings. Requires the ``item_embeddings`` table. A separate mode,
        not a fallback: it is the fully smooth alternative when graded
        similarity over the whole catalog is wanted.

    All modes return rewards with shape [B, num_candidates] (float32, in
    (0, 1]), floored at ``reward_epsilon``. ``sharpness`` raises the
    similarity to a power before the floor - similarity rewards are mild
    (log R std ~0.1-0.2), so gamma > 1 is the knob that makes them audible
    next to a dense collaborative reward in a product composition.
    """

    reward_epsilon: float = 1e-6
    mode: SimilarityMode = "exact"

    # batch key for the correct semantic target sequence
    target_token_key: str = "target_token_ids"
    # batch key for the positive target item id (embedding_cosine mode)
    target_item_key: str = "target_item_ids"

    # hierarchical: per-level weights, None -> equal. The notebook's prefix
    # analysis favors EQUAL weights: the gate already encodes the hierarchy,
    # and deep matches are rare and strong evidence.
    level_weights: tuple[float, ...] | None = None
    # semantic tokens are usually offset by one vs raw codebook indices
    # (token 0 is reserved for padding)
    token_offset: int = 1

    # exponent applied to the similarity before the epsilon floor
    sharpness: float = 1.0


class SimilarityRewardModel(nn.Module):
    """Similarity-to-target reward over semantic-ID candidates.

    Mode-specific assets (see :class:`SimilarityRewardConfig`):

    - ``codebooks``: RQ-VAE codebook tensor [n_levels, codebook_size, dim],
      required for ``mode="hierarchical"`` (see :func:`load_rqvae_codebooks`).
    - ``item_embeddings``: content-embedding table [item_vocab_size, dim]
      indexed by internal item id, required for ``mode="embedding_cosine"``
      (see :func:`build_item_embedding_table`). Normalized internally.
    - ``item_id_by_semantic_row``: optional row-id -> item-id mapping used to
      resolve candidates in ``embedding_cosine`` mode when trajectories carry
      semantic row ids instead of ``extra['candidate_item_ids']``.
    """

    def __init__(
        self,
        cfg: SimilarityRewardConfig | None = None,
        *,
        codebooks: torch.Tensor | None = None,
        item_embeddings: torch.Tensor | None = None,
        item_id_by_semantic_row: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SimilarityRewardConfig()

        if self.cfg.mode == "hierarchical":
            if codebooks is None:
                raise ValueError(
                    "mode='hierarchical' requires the RQ-VAE `codebooks` tensor "
                    "([n_levels, codebook_size, dim]); load it with "
                    "load_rqvae_codebooks(<rqvae model.pt>)."
                )
            if codebooks.ndim != 3:
                raise ValueError(
                    f"codebooks must have shape [n_levels, codebook_size, dim], "
                    f"got {tuple(codebooks.shape)}"
                )
            # Pairwise codebook-vector distances per level, normalized by the
            # level's max pairwise distance (Gower's "feature range" carried
            # into the latent metric space).
            dist = torch.cdist(codebooks.float(), codebooks.float())
            dist = dist / dist.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            self.register_buffer("code_dist", dist)
        else:
            self.register_buffer("code_dist", torch.empty(0))

        if self.cfg.mode == "embedding_cosine":
            if item_embeddings is None:
                raise ValueError(
                    "mode='embedding_cosine' requires the `item_embeddings` table "
                    "([item_vocab_size, dim]); build it with "
                    "build_item_embedding_table(...)."
                )
            if item_embeddings.ndim != 2:
                raise ValueError(
                    f"item_embeddings must have shape [item_vocab_size, dim], "
                    f"got {tuple(item_embeddings.shape)}"
                )
            self.register_buffer(
                "item_embeddings",
                nn.functional.normalize(item_embeddings.float(), dim=1),
            )
        else:
            self.register_buffer("item_embeddings", torch.empty(0))

        if item_id_by_semantic_row is None:
            self.register_buffer("item_id_by_semantic_row", torch.empty(0, dtype=torch.long))
        else:
            self.register_buffer("item_id_by_semantic_row", item_id_by_semantic_row.long())

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        if self.cfg.mode == "exact":
            sim = self._exact_similarity(batch, trajectories)
        elif self.cfg.mode == "hierarchical":
            sim = self._hierarchical_similarity(batch, trajectories)
        elif self.cfg.mode == "embedding_cosine":
            sim = self._embedding_similarity(batch, trajectories)
        else:
            raise ValueError(f"Unknown similarity mode: {self.cfg.mode!r}")

        if float(self.cfg.sharpness) != 1.0:
            sim = sim.clamp_min(0.0).pow(float(self.cfg.sharpness))
        return sim.clamp_min(float(self.cfg.reward_epsilon))

    # ------------------------------------------------------------------ #
    # exact: positionwise token agreement (legacy behavior)
    # ------------------------------------------------------------------ #
    def _candidate_and_target_tokens(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.cfg.target_token_key not in batch:
            raise KeyError(
                f"Batch is missing {self.cfg.target_token_key!r}; cannot compute similarity reward."
            )
        target = batch[self.cfg.target_token_key]
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)

        cand = trajectories.candidate_token_ids
        if cand.ndim != 3:
            raise ValueError(
                "candidate_token_ids must have shape [B, num_candidates, L], "
                f"got {tuple(cand.shape)}"
            )
        bsz, num_candidates, seq_len = cand.shape
        if target.ndim != 2 or target.shape[0] != bsz or target.shape[1] != seq_len:
            raise ValueError(
                "target_token_ids must have shape [B, L] matching trajectories. "
                f"target={tuple(target.shape)} cand={tuple(cand.shape)}"
            )
        target = target.to(cand.device).long()

        mask = getattr(trajectories, "candidate_token_mask", None)
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask)
            if mask.shape != (bsz, num_candidates, seq_len):
                raise ValueError(
                    "candidate_token_mask must have shape [B, num_candidates, L]; "
                    f"got {tuple(mask.shape)}"
                )
            mask = mask.to(cand.device).bool()
        return cand.long(), target, mask

    def _exact_similarity(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        cand, target, mask = self._candidate_and_target_tokens(batch, trajectories)
        eq = cand == target.unsqueeze(1)
        if mask is not None:
            eq = eq & mask
        return eq.sum(dim=-1).to(torch.float32) / cand.shape[-1]

    # ------------------------------------------------------------------ #
    # hierarchical: prefix-gated, codebook geometry at the first divergence
    # ------------------------------------------------------------------ #
    def _hierarchical_similarity(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        cand, target, mask = self._candidate_and_target_tokens(batch, trajectories)
        bsz, num_candidates, seq_len = cand.shape
        n_rq_levels, codebook_size = self.code_dist.shape[0], self.code_dist.shape[1]

        if self.cfg.level_weights is None:
            weights = torch.ones(seq_len, dtype=torch.float32, device=cand.device)
        else:
            if len(self.cfg.level_weights) != seq_len:
                raise ValueError(
                    f"level_weights has {len(self.cfg.level_weights)} entries but "
                    f"candidates carry {seq_len} token levels."
                )
            weights = torch.as_tensor(
                self.cfg.level_weights, dtype=torch.float32, device=cand.device
            )

        offset = int(self.cfg.token_offset)
        cand_codes = cand - offset
        target_codes = target - offset

        sim = torch.zeros((bsz, num_candidates), dtype=torch.float32, device=cand.device)
        active = torch.ones((bsz, num_candidates), dtype=torch.float32, device=cand.device)
        for level in range(seq_len):
            ca = cand_codes[:, :, level]
            ct = target_codes[:, level].unsqueeze(1)
            valid = (ca >= 0) & (ct >= 0)
            if mask is not None:
                valid = valid & mask[:, :, level]
            eq = (ca == ct) & valid

            if level < n_rq_levels:
                safe_ca = ca.clamp(0, codebook_size - 1)
                safe_ct = ct.clamp(0, codebook_size - 1).expand_as(safe_ca)
                dist = self.code_dist[level][safe_ca, safe_ct]
                grade = torch.where(eq, torch.ones_like(dist), 1.0 - dist)
                grade = grade * valid.float()
            else:
                grade = eq.float()

            sim = sim + weights[level] * active * grade
            active = active * eq.float()

        return sim / weights.sum().clamp_min(1e-12)

    # ------------------------------------------------------------------ #
    # embedding_cosine: content-embedding similarity, no quantization
    # ------------------------------------------------------------------ #
    def _embedding_similarity(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        candidate_item_ids = self._resolve_candidate_item_ids(batch, trajectories)
        vocab_size = self.item_embeddings.shape[0]
        invalid = candidate_item_ids.lt(0) | candidate_item_ids.ge(vocab_size)
        safe_ids = candidate_item_ids.clamp(0, vocab_size - 1)

        target_item_ids = batch.get(self.cfg.target_item_key)
        if not torch.is_tensor(target_item_ids):
            raise KeyError(
                f"Batch is missing {self.cfg.target_item_key!r}; cannot compute "
                "embedding_cosine similarity."
            )
        target_item_ids = target_item_ids.to(safe_ids.device).long()
        target_vecs = self.item_embeddings[target_item_ids.clamp(0, vocab_size - 1)]

        # Cosine via the full-vocab table [V, B] instead of gathering [B, N, D]
        # candidate vectors - keeps memory flat regardless of candidate count.
        all_cos = self.item_embeddings @ target_vecs.T                 # [V, B]
        batch_idx = torch.arange(safe_ids.shape[0], device=safe_ids.device).unsqueeze(1)
        cos = all_cos[safe_ids, batch_idx]                             # [B, N]
        sim = (cos + 1.0) / 2.0
        return sim.masked_fill(invalid, 0.0)

    def _resolve_candidate_item_ids(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        """Resolve candidates to internal atomic item ids (-1 = unresolved)."""
        device = trajectories.candidate_token_ids.device
        bsz = trajectories.batch_size
        n_candidates = trajectories.num_candidates

        from_extra = trajectories.extra.get("candidate_item_ids")
        if torch.is_tensor(from_extra):
            candidate_item_ids = from_extra.to(device).long()
        elif trajectories.candidate_item_row_ids is not None:
            candidate_item_ids = trajectories.candidate_item_row_ids.to(device).long()
            if self.item_id_by_semantic_row.numel() > 0:
                mapping = self.item_id_by_semantic_row.to(device)
                valid_rows = (candidate_item_ids >= 0) & (candidate_item_ids < mapping.numel())
                safe_rows = candidate_item_ids.clamp(0, mapping.numel() - 1)
                mapped = mapping[safe_rows]
                candidate_item_ids = torch.where(valid_rows, mapped, torch.full_like(mapped, -1))
        else:
            candidate_item_ids = torch.full(
                (bsz, n_candidates), -1, device=device, dtype=torch.long
            )

        if candidate_item_ids.shape != (bsz, n_candidates):
            raise ValueError(
                "candidate_item_ids must have shape [B, N], got "
                f"{tuple(candidate_item_ids.shape)} expected {(bsz, n_candidates)}"
            )

        target_item_ids = batch.get(self.cfg.target_item_key)
        if torch.is_tensor(target_item_ids):
            target_item_ids = target_item_ids.to(device).long().reshape(bsz, 1)
            candidate_item_ids = torch.where(
                trajectories.is_positive.to(device).bool(),
                target_item_ids.expand_as(candidate_item_ids),
                candidate_item_ids,
            )
        return candidate_item_ids


# ---------------------------------------------------------------------- #
# asset helpers
# ---------------------------------------------------------------------- #
def load_rqvae_codebooks(model_path: str | Path) -> torch.Tensor:
    """Load the [n_levels, codebook_size, dim] codebook tensor from an RQ-VAE
    checkpoint (the ``model.pt`` written next to ``semantic_ids.json``)."""
    payload = torch.load(Path(model_path), map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if "quantizer.codebooks" not in state_dict:
        raise KeyError(
            f"{model_path} does not contain 'quantizer.codebooks'; is it an RQ-VAE checkpoint?"
        )
    return state_dict["quantizer.codebooks"].float()


def build_item_embedding_table(
    embeddings_path: str | Path,
    item2id: Mapping[str, int],
    item_vocab_size: int,
) -> torch.Tensor:
    """Build a [item_vocab_size, dim] content-embedding table indexed by
    internal item id from a ``*_content_embeddings.pt`` payload
    (``{"item_ids": [...], "embeddings": Tensor}``). Items without an
    embedding keep a zero row (their cosine similarity is 0 -> reward floor).
    """
    payload = torch.load(Path(embeddings_path), map_location="cpu", weights_only=False)
    raw_ids = [str(x) for x in payload["item_ids"]]
    embeddings = payload["embeddings"].float()
    if embeddings.ndim != 2 or len(raw_ids) != embeddings.shape[0]:
        raise ValueError(f"Malformed embeddings payload at {embeddings_path}")

    table = torch.zeros((item_vocab_size, embeddings.shape[1]), dtype=torch.float32)
    hits = 0
    for row, raw in enumerate(raw_ids):
        internal = item2id.get(raw)
        if internal is not None and 0 <= int(internal) < item_vocab_size:
            table[int(internal)] = embeddings[row]
            hits += 1
    if hits == 0:
        raise ValueError(
            f"No embedding item_ids from {embeddings_path} matched the item catalog."
        )
    return table
