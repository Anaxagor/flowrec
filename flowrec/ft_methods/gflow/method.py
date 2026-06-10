from __future__ import annotations

from typing import Any

import torch

from flowrec.candidates.generated import GeneratedSemanticCandidateConfig, GeneratedSemanticIDCandidateSampler
from flowrec.candidates.random import RandomSemanticCandidateConfig, RandomSemanticIDCandidateSampler
from flowrec.ft_methods.base import FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.flow import (
    GlobalLogZ,
    SemanticHistoryLogZ,
    SemanticHistoryLogZConfig,
    VectorLogZ,
)
#from flowrec.ft_methods.gflow.losses.detailed_balance import detailed_balance_loss
from flowrec.ft_methods.gflow.losses.trajectory_balance import trajectory_balance_loss
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import Batch, FlowRecModel
from flowrec.rewards.rating import RatingRewardConfig, RatingRewardModel


class GFlowMethod(FlowRecMethod):
    """GFlowGR fine-tuning with rating rewards and trajectory balance.

    The method follows the paper's clean GR specialization:
      - SFT keeps the observed next-item supervision.
      - Candidate semantic-ID trajectories receive terminal rewards.
      - The TIGER model supplies forward token probabilities.
      - Backward probabilities are one for left-to-right generation.
    """

    method_name = "gflow"

    def __init__(
        self,
        cfg: GFlowConfig | None = None,
        *,
        candidate_sampler: RandomSemanticIDCandidateSampler | None = None,
        reward_model: RatingRewardModel | None = None,
        sft_method: SFTMethod | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or GFlowConfig()
        self.log_z_model = self._build_log_z_model()

        self.random_candidate_sampler = RandomSemanticIDCandidateSampler(
            RandomSemanticCandidateConfig(
                num_negative_samples=self.cfg.num_negative_samples,
                include_positive=self.cfg.include_positive,
                avoid_positive=self.cfg.avoid_positive,
                pad_token_id=self.cfg.pad_token_id,
                semantic_ids_are_fixed_length=self.cfg.semantic_ids_are_fixed_length,
                max_resample_attempts=self.cfg.max_resample_attempts,
            )
        )
        self.generated_candidate_sampler = GeneratedSemanticIDCandidateSampler(
            GeneratedSemanticCandidateConfig(
                num_negative_samples=self.cfg.num_negative_samples,
                include_positive=self.cfg.include_positive,
                num_beams=self.cfg.generated_num_beams,
                num_return_sequences=self.cfg.generated_num_return_sequences,
                oversample_factor=self.cfg.generated_oversample_factor,
                constrain_valid_sequences=self.cfg.generated_constrain_valid_sequences,
                exclude_seen_items=self.cfg.generated_exclude_seen_items,
                pad_token_id=self.cfg.pad_token_id,
                semantic_ids_are_fixed_length=self.cfg.semantic_ids_are_fixed_length,
                max_resample_attempts=self.cfg.max_resample_attempts,
            )
        )
        self.candidate_sampler = candidate_sampler
        self.reward_model = reward_model or RatingRewardModel(
            RatingRewardConfig(
                reward_key=self.cfg.reward_key,
                reward_epsilon=self.cfg.reward_epsilon,
                negative_reward=self.cfg.negative_reward,
                transform=self.cfg.reward_transform,
            )
        )
        self.sft_method = sft_method or SFTMethod(
            SFTConfig(
                compute_train_metrics=self.cfg.compute_train_metrics,
                compute_eval_metrics=self.cfg.compute_eval_metrics,
                compute_token_metrics=self.cfg.compute_token_metrics,
                compute_item_metrics=False,
                evaluate_generation=self.cfg.evaluate_generation,
                generation_topk=self.cfg.generation_topk,
                generation_num_beams=self.cfg.generation_num_beams,
                generation_num_return_sequences=self.cfg.generation_num_return_sequences,
                generation_constrain_valid_sequences=self.cfg.generation_constrain_valid_sequences,
                generation_exclude_seen_items=self.cfg.generation_exclude_seen_items,
            )
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
        trajectories = self._sample_trajectories(model, batch, step=step)
        rewards = self.reward_model(batch, trajectories)

        sequence_logprobs = model.score_items(
            batch,
            candidate_token_ids=trajectories.candidate_token_ids,
            candidate_token_mask=trajectories.candidate_token_mask,
        )
        log_z = self.log_z_model(batch)

        if self.cfg.objective == "tb":
            gflow_loss, gflow_metrics = trajectory_balance_loss(
                sequence_logprobs=sequence_logprobs,
                rewards=rewards,
                log_z=log_z,
                reward_epsilon=self.cfg.reward_epsilon,
            )
        elif self.cfg.objective == "db":
            gflow_loss, gflow_metrics = detailed_balance_loss()
        else:
            raise ValueError(f"Unknown GFlow objective: {self.cfg.objective}")

        loss = float(self.cfg.sft_weight) * sft_out.loss + float(self.cfg.gflow_weight) * gflow_loss
        metrics: dict[str, Any] = {
            "loss": loss.detach(),
            "sft_loss": sft_out.loss.detach(),
            "gflow_loss": gflow_loss.detach(),
            "log_z_mean": log_z.detach().mean(),
            "log_z_std": log_z.detach().reshape(-1).std(unbiased=False),
            "num_candidates": trajectories.candidate_token_ids.new_tensor(
                trajectories.num_candidates, dtype=torch.float32
            ),
            "num_negative_samples": trajectories.candidate_token_ids.new_tensor(
                max(trajectories.num_candidates - int(self.cfg.include_positive), 0),
                dtype=torch.float32,
            ),
            "candidate_unique_ratio": self._candidate_unique_ratio(trajectories.candidate_token_ids),
            "generated_sampler_active": trajectories.candidate_token_ids.new_tensor(
                1.0 if trajectories.extra.get("sampler") == "generated_semantic_id" else 0.0,
                dtype=torch.float32,
            ),
            "score_sampler_active": trajectories.candidate_token_ids.new_tensor(
                1.0 if str(trajectories.extra.get("sampler", "")).endswith("_semantic_id")
                and trajectories.extra.get("sampler") not in {"random_semantic_id", "generated_semantic_id"}
                else 0.0,
                dtype=torch.float32,
            ),
            "easy_negative_active": trajectories.candidate_token_ids.new_tensor(
                1.0 if trajectories.extra.get("sampler_phase") == "easy" else 0.0,
                dtype=torch.float32,
            ),
        }
        metrics.update(gflow_metrics)

        method_metrics = prefix_metrics(metrics, prefix=self.cfg.metric_prefix)
        method_metrics.update(sft_out.metrics)

        extra = dict(sft_out.extra)
        extra.update(
            {
                "stage": stage,
                "step": step,
                "trajectories": trajectories,
                "rewards": rewards,
                "sequence_logprobs": sequence_logprobs,
            }
        )

        return MethodOutput(
            loss=loss,
            metrics=method_metrics,
            model_output=sft_out.model_output,
            extra=extra,
        )

    def _build_log_z_model(self) -> torch.nn.Module:
        if self.cfg.logz_context == "global":
            return GlobalLogZ(init_value=self.cfg.logz_init)
        if self.cfg.logz_context == "vector":
            return VectorLogZ(
                vector_size=int(self.cfg.logz_vector_size),
                init_value=float(self.cfg.logz_init),
            )
        if self.cfg.logz_context == "semantic_history":
            if self.cfg.logz_vocab_size is None:
                raise ValueError(
                    "GFlowConfig.logz_vocab_size is required when "
                    "logz_context='semantic_history'."
                )
            return SemanticHistoryLogZ(
                SemanticHistoryLogZConfig(
                    vocab_size=int(self.cfg.logz_vocab_size),
                    embedding_dim=int(self.cfg.logz_embedding_dim),
                    hidden_dim=int(self.cfg.logz_hidden_dim),
                    dropout=float(self.cfg.logz_dropout),
                    pad_token_id=int(self.cfg.pad_token_id),
                    init_value=float(self.cfg.logz_init),
                )
            )
        raise ValueError(f"Unknown logZ context: {self.cfg.logz_context}")

    def _sample_trajectories(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        step: int | None,
    ):
        if self.candidate_sampler is not None:
            return self.candidate_sampler.sample(batch, model=model, step=step)

        if self.cfg.candidate_sampler == "random":
            return self.random_candidate_sampler.sample(batch, model=model)

        if self.cfg.candidate_sampler == "generated":
            refresh = max(int(self.cfg.generated_refresh_every_steps), 1)
            if step is not None and step % refresh != 0:
                trajectories = self.random_candidate_sampler.sample(batch, model=model)
                trajectories.extra["sampler"] = "random_semantic_id_scheduled"
                return trajectories
            return self.generated_candidate_sampler.sample(batch, model=model)

        raise ValueError(f"Unknown candidate sampler: {self.cfg.candidate_sampler}")

    def _candidate_unique_ratio(self, candidate_token_ids: torch.Tensor) -> torch.Tensor:
        flat = candidate_token_ids.reshape(-1, candidate_token_ids.size(-1))
        if flat.numel() == 0:
            return torch.zeros((), device=candidate_token_ids.device)
        unique_count = torch.unique(flat, dim=0).size(0)
        return torch.tensor(
            unique_count / max(flat.size(0), 1),
            device=candidate_token_ids.device,
            dtype=torch.float32,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GFlowMethod":
        return cls(cfg=GFlowConfig.from_dict(data))
