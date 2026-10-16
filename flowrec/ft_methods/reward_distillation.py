"""Cross-entropy distillation from an exact full-catalog reward distribution.

The teacher is ``q(i | h) = R(h, i) / sum_catalog R(h, i)``. Its CE with the
student is added to the usual SFT anchor. The student uses the same locally
normalized valid catalog actions as GFlow, summing token log-probabilities
along an item's semantic ID (not averaging over its tokens).

``sampled`` draws IID catalog rows with replacement. Direct reward sampling
uses q; the two-phase curriculum first conditions q on the lower-reward half
of the catalog and then on its higher-reward half. A separate smooth mode
linearly mixes the two conditional distributions during the transition.
Both estimate the active conditional (mixture) CE without importance correction.
``curriculum_smooth_is`` adds a positive q component to that proposal, preserving
q's support even at the endpoints, and applies raw q/proposal importance weights
to estimate the original full-catalog CE without bias.
No sampling mode uses candidate-set normalization, history exclusion or positive injection.
``exact`` aggregates q onto catalog prefix edges and scores each branching
prefix once. Deterministic next actions contribute zero. Decoder chunks are
checkpointed to bound saved activations. TIGER reuses SFT's encoder; GPT2
shares one differentiable history prefill across CE candidates and chunks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import torch
from torch.utils.checkpoint import checkpoint

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY, FlowRecMethod, MethodOutput
from flowrec.ft_methods.gflow.prefetch import PREPARED_REWARDS_KEY
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import Batch, FlowRecModel


@dataclass(frozen=True, slots=True)
class RewardDistillationConfig:
    name: str = "reward_distillation"
    estimator: Literal["sampled", "exact"] = "sampled"
    num_samples: int = 10
    catalog_chunk_size: int = 16
    distill_weight: float = 1.0
    normalization: str = "valid_actions"
    sampling: Literal["reward", "curriculum", "curriculum_smooth", "curriculum_smooth_is"] = "reward"
    curriculum_steps: int = 0
    exact_algorithm: str = "branching_prefix"
    proposal_q_mix: float = 0.1

    def __post_init__(self) -> None:
        if self.estimator not in {"sampled", "exact"}:
            raise ValueError("estimator must be 'sampled' or 'exact'.")
        if self.num_samples < 1 or int(self.num_samples) != self.num_samples:
            raise ValueError("num_samples must be a positive integer.")
        if self.catalog_chunk_size < 1 or int(self.catalog_chunk_size) != self.catalog_chunk_size:
            raise ValueError("catalog_chunk_size must be a positive integer.")
        if not math.isfinite(self.distill_weight) or self.distill_weight < 0:
            raise ValueError("distill_weight must be finite and nonnegative.")
        if self.normalization != "valid_actions":
            raise ValueError("Reward distillation requires valid_actions normalization.")
        if self.sampling not in {"reward", "curriculum", "curriculum_smooth", "curriculum_smooth_is"}:
            raise ValueError("sampling must be 'reward', 'curriculum', 'curriculum_smooth' or 'curriculum_smooth_is'.")
        if not math.isfinite(self.proposal_q_mix) or not 0 < self.proposal_q_mix <= 1:
            raise ValueError("proposal_q_mix must be finite and in (0, 1].")
        if (self.sampling == "curriculum_smooth_is"
                and self.proposal_q_mix < 1.0 / torch.finfo(torch.float32).max):
            raise ValueError("proposal_q_mix is too small: importance weights must be representable in float32.")
        if self.curriculum_steps < 0 or int(self.curriculum_steps) != self.curriculum_steps:
            raise ValueError("curriculum_steps must be a nonnegative integer.")
        if self.estimator == "exact" and self.sampling != "reward":
            raise ValueError("A curriculum applies only to the sampled estimator.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Keep serialized configurations of existing experiments unchanged.
        if self.sampling != "curriculum_smooth_is":
            result.pop("proposal_q_mix")
        return result


class RewardDistillationMethod(FlowRecMethod):
    """Generative SFT + reward CE, with optional reward-half curricula.

    Catalog row order must match the cache. The catalog includes seen items:
    inference-time seen-item filtering does not alter the training teacher.
    Validation/test delegate directly to SFT and never access reward caches.
    """

    method_name = "reward_distillation"

    def __init__(
        self,
        reward_cache: Any,
        item_semantic_ids: torch.Tensor,
        sft_method: SFTMethod,
        *,
        estimator: Literal["sampled", "exact"] = "sampled",
        num_samples: int = 10,
        catalog_chunk_size: int = 16,
        distill_weight: float = 1.0,
        sampling: Literal["reward", "curriculum", "curriculum_smooth", "curriculum_smooth_is"] = "reward",
        curriculum_steps: int = 0,
        proposal_q_mix: float = 0.1,
    ) -> None:
        super().__init__()
        self.cfg = RewardDistillationConfig(
            estimator=estimator,
            num_samples=num_samples,
            catalog_chunk_size=catalog_chunk_size,
            distill_weight=float(distill_weight),
            sampling=sampling,
            curriculum_steps=curriculum_steps,
            proposal_q_mix=float(proposal_q_mix),
        )
        if item_semantic_ids.ndim != 2 or min(item_semantic_ids.shape) == 0:
            raise ValueError("item_semantic_ids must be a nonempty [catalog_size, sid_length] tensor.")
        if item_semantic_ids.is_floating_point() or item_semantic_ids.is_complex():
            raise ValueError("item_semantic_ids must contain integer token IDs.")
        if bool(item_semantic_ids.lt(0).any()):
            raise ValueError("Catalog semantic IDs must not contain padding or negative tokens.")
        self.register_buffer("item_semantic_ids", item_semantic_ids.detach().long(), persistent=False)
        self.reward_cache = reward_cache
        self.sft_method = sft_method
        self._prefix_levels: list[dict[str, torch.Tensor]] | None = None
        self._prefix_device: torch.device | None = None

    @property
    def distill_weight(self) -> float:
        return self.cfg.distill_weight

    @distill_weight.setter
    def distill_weight(self, value: float) -> None:
        self.cfg = replace(self.cfg, distill_weight=float(value))

    def curriculum_high_probability(self, step: int | None) -> float:
        """Scheduled high-half mixture weight for either curriculum mode."""
        if self.cfg.curriculum_steps == 0:
            return 1.0
        current_step = 0 if step is None else int(step)
        if self.cfg.sampling in {"curriculum_smooth", "curriculum_smooth_is"}:
            return min(max(current_step / self.cfg.curriculum_steps, 0.0), 1.0)
        return float(current_step >= self.cfg.curriculum_steps)

    def curriculum_phase(self, step: int | None) -> str:
        """An omitted step starts at 0; smooth transitions add a mixed phase."""
        if self.cfg.sampling == "reward":
            return "all"
        high_probability = self.curriculum_high_probability(step)
        if high_probability == 0.0:
            return "low"
        return "high" if high_probability == 1.0 else "mixed"

    def sampling_distribution(self, q: torch.Tensor, *, step: int | None) -> tuple[torch.Tensor, str]:
        phase = self.curriculum_phase(step)
        importance_corrected = self.cfg.sampling == "curriculum_smooth_is"
        if importance_corrected:
            q = q.detach()
            # Pure q needs no conditional distributions: one of the halves
            # can legitimately have zero mass when the teacher has zeros.
            if self.cfg.proposal_q_mix == 1.0:
                return q, phase
        if phase == "all":
            return q, phase
        size = q.size(1)
        if size == 1:
            return q, phase
        # q has the same per-context ordering as the composed rewards. Ties
        # keep catalog order, making the floor(N/2) high half reproducible.
        order = torch.argsort(q, dim=1, descending=True, stable=True)
        split = size // 2

        def conditional(selected: torch.Tensor) -> torch.Tensor:
            probabilities = torch.zeros_like(q)
            probabilities.scatter_(1, selected, q.gather(1, selected))
            selected_mass = probabilities.sum(dim=1, keepdim=True)
            if not bool(selected_mass.gt(0).all()):
                raise ValueError("The active curriculum half must have positive teacher reward mass.")
            return probabilities / selected_mass

        # At either endpoint only its active half needs positive mass. In a
        # mixed phase each conditional is normalized independently: alpha is
        # the scheduled mass of the high half, regardless of its original q mass.
        if phase == "low":
            proposal = conditional(order[:, split:])
        elif phase == "high":
            proposal = conditional(order[:, :split])
        else:
            alpha = self.curriculum_high_probability(step)
            proposal = ((1.0 - alpha) * conditional(order[:, split:])
                        + alpha * conditional(order[:, :split]))
        if importance_corrected:
            epsilon = self.cfg.proposal_q_mix
            proposal = (1.0 - epsilon) * proposal + epsilon * q
            if bool((q.gt(0) & (proposal.le(0) | ~torch.isfinite(proposal))).any()):
                raise ValueError("proposal_q_mix cannot preserve full q support at this precision; use a larger value.")
        return proposal, phase

    def _branching_prefix_levels(self, device: torch.device) -> list[dict[str, torch.Tensor]]:
        if self._prefix_levels is None:
            catalog = self.item_semantic_ids.detach().cpu().tolist()
            levels: list[dict[str, torch.Tensor]] = []
            for depth in range(len(catalog[0])):
                groups: dict[tuple[int, ...], dict[int, list[int]]] = {}
                for row, tokens in enumerate(catalog):
                    children = groups.setdefault(tuple(tokens[:depth]), {})
                    children.setdefault(tokens[depth], []).append(row)
                nodes = [(prefix, children) for prefix, children in groups.items() if len(children) > 1]
                if not nodes:
                    continue
                width = max(len(children) for _, children in nodes)
                tokens = torch.empty((len(nodes), depth + 1), dtype=torch.long)
                child_ids = torch.zeros((len(nodes), width), dtype=torch.long)
                child_mask = torch.zeros((len(nodes), width), dtype=torch.bool)
                child_edges = torch.zeros((len(nodes), width), dtype=torch.long)
                source_rows: list[int] = []
                edge_indices: list[int] = []
                edge = 0
                for node, (prefix, children) in enumerate(nodes):
                    ordered_children = sorted(children)
                    # The last token is a placeholder: right shifting means
                    # last-step logits depend only on the prefix before it.
                    tokens[node] = torch.tensor((*prefix, ordered_children[0]))
                    for slot, child in enumerate(ordered_children):
                        child_ids[node, slot] = child
                        child_mask[node, slot] = True
                        child_edges[node, slot] = edge
                        source_rows.extend(children[child])
                        edge_indices.extend([edge] * len(children[child]))
                        edge += 1
                levels.append({
                    "tokens": tokens, "child_ids": child_ids, "child_mask": child_mask,
                    "child_edges": child_edges,
                    "source_rows": torch.tensor(source_rows, dtype=torch.long),
                    "edge_indices": torch.tensor(edge_indices, dtype=torch.long),
                })
            self._prefix_levels = levels
        if self._prefix_device != device:
            self._prefix_levels = [
                {key: value.to(device) for key, value in level.items()}
                for level in self._prefix_levels
            ]
            self._prefix_device = device
        return self._prefix_levels

    def _exact_prefix_ce(
        self, model: FlowRecModel, q: torch.Tensor, scoring_batch: Batch, zero: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        scorer = getattr(model, "candidate_token_logits", None)
        if not callable(scorer):
            raise TypeError("Exact reward distillation requires model.candidate_token_logits().")

        def weighted_prefix_chunk(tokens, masses, child_ids, child_mask, context_batch):
            logits = scorer(
                context_batch, candidate_token_ids=tokens,
                candidate_token_mask=torch.ones_like(tokens, dtype=torch.bool),
            )[..., -1, :].float()
            child_logits = logits.gather(-1, child_ids.unsqueeze(0).expand(q.size(0), -1, -1))
            mask = child_mask.unsqueeze(0)
            log_probs = child_logits.masked_fill(~mask, -torch.inf).log_softmax(-1).masked_fill(~mask, 0.)
            return -(masses * log_probs).sum(dim=(1, 2)).mean()

        # A one-leaf catalog has CE zero; retain a valid zero autograd path.
        loss = zero
        prefix_count = 0
        for level in self._branching_prefix_levels(q.device):
            prefix_count += len(level["tokens"])
            num_edges = int(level["edge_indices"].max().item()) + 1
            masses = q.new_zeros((q.size(0), num_edges))
            masses.scatter_add_(
                1, level["edge_indices"].unsqueeze(0).expand(q.size(0), -1),
                q.index_select(1, level["source_rows"]),
            )
            for start in range(0, len(level["tokens"]), self.cfg.catalog_chunk_size):
                end = start + self.cfg.catalog_chunk_size
                mask = level["child_mask"][start:end]
                chunk_masses = masses[:, level["child_edges"][start:end]].masked_fill(~mask, 0.)
                tokens = level["tokens"][start:end].unsqueeze(0).expand(q.size(0), -1, -1)
                # Non-reentrant checkpoint supports nested tensor arguments,
                # including GPT2's differentiable K/V tuples in context_batch.
                args = (tokens, chunk_masses, level["child_ids"][start:end], mask, scoring_batch)
                value = (checkpoint(weighted_prefix_chunk, *args, use_reentrant=False)
                         if torch.is_grad_enabled() else weighted_prefix_chunk(*args))
                loss = loss + value
        return loss, prefix_count

    def _teacher_distribution(self, batch: Batch, device: torch.device) -> torch.Tensor:
        indices = batch.get("example_indices")
        if not torch.is_tensor(indices) or indices.ndim != 1:
            raise ValueError("Reward distillation requires batch['example_indices'] with shape [B].")
        empty_rows = torch.empty((indices.numel(), 0), dtype=torch.long, device=device)
        prepared = batch.get(PREPARED_REWARDS_KEY)
        # Teacher computation must never add a trainable path through a cache.
        with torch.no_grad():
            _, sums, full = self.reward_cache.select_candidates(
                indices,
                empty_rows,
                device=device,
                include_full=True,
                **({"prepared": prepared} if prepared is not None else {}),
            )
            if not torch.is_tensor(full) or full.shape != (indices.numel(), self.item_semantic_ids.size(0)):
                raise ValueError("Full reward rows must match the batch and the complete semantic-ID catalog.")
            full = full.detach().to(device=device, dtype=torch.float32)
            sums = sums.detach().to(device=device, dtype=torch.float32)
            if sums.shape != (indices.numel(),):
                raise ValueError("Reward sums must have shape [B].")
            if not bool(torch.isfinite(full).all() & full.ge(0).all()):
                raise ValueError("Full rewards must be finite and nonnegative.")
            if not bool(torch.isfinite(sums).all() & sums.gt(0).all()):
                raise ValueError("Full-catalog reward sums must be finite and positive.")
            if not torch.allclose(full.sum(dim=1), sums, rtol=1e-5, atol=1e-8):
                raise ValueError("Reward sums must normalize the same full-catalog reward rows.")
            return full / sums[:, None]

    @staticmethod
    def _scoring_batch(model: FlowRecModel, batch: Batch, sft_out: MethodOutput) -> dict[str, Any]:
        extra = getattr(sft_out.model_output, "extra", None)
        hidden = extra.get("encoder_last_hidden_state") if isinstance(extra, dict) else None
        mask = extra.get("encoder_attention_mask") if isinstance(extra, dict) else None
        if torch.is_tensor(hidden) and torch.is_tensor(mask):
            return {
                **batch,
                "_tiger_encoder_last_hidden_state": hidden,
                "_tiger_encoder_attention_mask": mask,
            }
        prepare = getattr(model, "cache_history_encoding_grad", None)
        if callable(prepare):
            return dict(prepare(batch))
        raise ValueError(
            "Reward distillation requires SFT encoder state or model.cache_history_encoding_grad()."
        )

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        sft_out = self.sft_method.compute_loss(model, batch, stage=stage, step=step)
        if stage != "train":
            return sft_out
        scorer = getattr(model, "score_item_tokens_valid_actions", None)
        if not callable(scorer):
            raise TypeError("Reward distillation requires model.score_item_tokens_valid_actions().")
        device = sft_out.loss.device
        q = self._teacher_distribution(batch, device)
        catalog = self.item_semantic_ids.to(device=device)
        scoring_batch = self._scoring_batch(model, batch, sft_out)

        def weighted_chunk(
            tokens: torch.Tensor, weights: torch.Tensor, context_batch: Batch
        ) -> torch.Tensor:
            token_lp = scorer(
                context_batch,
                candidate_token_ids=tokens,
                candidate_token_mask=torch.ones_like(tokens, dtype=torch.bool),
            )
            if token_lp.shape != tokens.shape:
                raise ValueError("Valid-action token scores must have shape [B, N, sid_length].")
            return -(weights * token_lp.float().sum(dim=-1)).sum(dim=1).mean()

        sampled_rows = None
        sampling_metrics: dict[str, Any] = {}
        if self.cfg.estimator == "sampled":
            proposal, phase = self.sampling_distribution(q, step=step)
            sampled_rows = torch.multinomial(proposal, self.cfg.num_samples, replacement=True)
            tokens = catalog[sampled_rows]
            importance_weights = None
            if self.cfg.sampling == "curriculum_smooth_is":
                # Raw importance weights, without clipping or normalization
                # over the samples: E_proposal[(q/proposal) * loss] = CE(q).
                importance_weights = q.gather(1, sampled_rows) / proposal.gather(1, sampled_rows)
                if not bool(torch.isfinite(importance_weights).all()):
                    raise ValueError("Non-finite importance weights; increase proposal_q_mix instead of clipping weights.")
                weights = importance_weights / self.cfg.num_samples
            else:
                weights = q.new_full(sampled_rows.shape, 1.0 / self.cfg.num_samples)
            distill_loss = weighted_chunk(tokens, weights, scoring_batch)
            if phase != "all":
                active = proposal.gt(0)
                sampling_metrics = {
                    "distill/curriculum_phase_high": float(phase == "high"),
                    "distill/curriculum_high_probability": self.curriculum_high_probability(step),
                    "distill/selected_teacher_mass": (q * active).sum(dim=1).mean(),
                    "distill/sampling_support_count": active.sum(dim=1).float().mean(),
                }
            if importance_weights is not None:
                # Diagnostics must not overflow when valid raw weights are
                # large. This higher-precision reduction never changes loss.
                diagnostic_weights = importance_weights.double()
                sampling_metrics.update({
                    "distill/proposal_q_mix": self.cfg.proposal_q_mix,
                    "distill/importance_weight_mean": diagnostic_weights.mean().to(q.dtype),
                    "distill/importance_weight_max": importance_weights.max(),
                    "distill/importance_ess": (diagnostic_weights.sum(dim=1).square()
                                              / diagnostic_weights.square().sum(dim=1)).mean().to(q.dtype),
                })
        else:
            distill_loss, prefix_count = self._exact_prefix_ce(model, q, scoring_batch, sft_out.loss.float() * 0.)
            sampling_metrics["distill/exact_branching_prefixes"] = prefix_count

        loss = sft_out.loss + self.distill_weight * distill_loss
        metrics = {
            **sft_out.metrics,
            **sampling_metrics,
            "distill/loss": distill_loss.detach(),
            "distill/total_loss": loss.detach(),
            "distill/weight": self.distill_weight,
            "distill/teacher_entropy": -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum(dim=1).mean(),
            "distill/teacher_max_prob": q.amax(dim=1).mean(),
        }
        targets = batch.get("target_token_ids")
        if torch.is_tensor(targets) and targets.shape == (q.size(0), catalog.size(1)):
            matches = (catalog[None, :, :] == targets.to(device)[:, None, :]).all(dim=-1)
            metrics["distill/teacher_target_prob"] = (q * matches).sum(dim=1).mean()
        extra = {
            **sft_out.extra,
            LOSS_COMPONENTS_KEY: {"sft": sft_out.loss, "distill": distill_loss},
        }
        if sampled_rows is not None:
            extra["distill_sampled_item_rows"] = sampled_rows
        return MethodOutput(loss=loss, metrics=metrics, model_output=sft_out.model_output, extra=extra)
