from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from ..base import SemanticIDStore, SemanticIDTokenizer, dataclass_to_json_dict
from .codebook import (
    CollisionPolicy,
    SemanticIDCollisionResolver,
    codes_to_mapping,
    offset_code_tokens,
)
from .rqvae import (
    RQVAE,
    RQVAEConfig,
    RQVAETrainConfig,
    RQVAETrainer,
    _normalize_train_config_dict,
    load_rqvae_checkpoint,
    save_rqvae_checkpoint,
)

NormalizeMode = Literal["none", "l2", "standard"]


@dataclass(frozen=True)
class RQVAEItemTokenizerConfig:
    rqvae: RQVAEConfig
    train: RQVAETrainConfig = RQVAETrainConfig()
    normalize: NormalizeMode = "none"
    collision_policy: CollisionPolicy = "append"
    level_offset_tokens: bool = True
    collision_token_offset: int | None = None
    collision_suffix_size: int | None = None
    collision_seed: int = 42
    encode_batch_size: int = 4096
    


class EmbeddingNormalizer:
    def __init__(self, mode: NormalizeMode = "l2") -> None:
        self.mode = mode
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "EmbeddingNormalizer":
        if self.mode == "standard":
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.mode == "none":
            return x
        if self.mode == "l2":
            return F.normalize(x, p=2, dim=1)
        if self.mode == "standard":
            if self.mean is None or self.std is None:
                raise RuntimeError("EmbeddingNormalizer(mode='standard') must be fitted first")
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        raise ValueError(f"Unknown normalization mode: {self.mode}")

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mean": self.mean,
            "std": self.std,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mode = state["mode"]
        self.mean = state.get("mean")
        self.std = state.get("std")


class RQVAEItemTokenizer(SemanticIDTokenizer):
    """Train RQ-VAE and encode dense embeddings into semantic IDs."""

    def __init__(self, config: RQVAEItemTokenizerConfig) -> None:
        self.config = config
        self.normalizer = EmbeddingNormalizer(config.normalize)
        self.model = RQVAE(config.rqvae)
        self.train_state = None
        self.collision_stats = None
        self.semantic_id_store: SemanticIDStore | None = None

    def fit(self, embeddings: torch.Tensor, item_ids: Sequence[str] | None = None) -> "RQVAEItemTokenizer":
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [num_items, embedding_dim]")
        if embeddings.shape[1] != self.config.rqvae.input_dim:
            raise ValueError(
                f"RQVAE input_dim={self.config.rqvae.input_dim} but embeddings dim={embeddings.shape[1]}"
            )
        x = self.normalizer.fit_transform(embeddings.float())
        trainer = RQVAETrainer(self.model, self.config.train)
        self.train_state = trainer.fit(x)

        if item_ids is not None:
            codes = self._prepare_export_codes(self.encode(embeddings))
            resolver = SemanticIDCollisionResolver(
                self.config.collision_policy,
                suffix_offset=self._collision_suffix_offset(),
                suffix_size=self._collision_suffix_size(),
                seed=self.config.collision_seed,
            )
            resolved, stats = resolver.resolve(item_ids, codes)
            self.collision_stats = stats
            mapping = codes_to_mapping(item_ids, resolved)
            self.semantic_id_store = SemanticIDStore(
                mapping,
                metadata={
                    "tokenizer": "rqvae",
                    "normalize": self.config.normalize,
                    "collision_stats": stats.to_dict(),
                    "codebook_init": self.config.train.codebook_init,
                    "level_offset_tokens": self.config.level_offset_tokens,
                    "collision_token_offset": self._collision_suffix_offset(),
                    "rqvae_config": dataclass_to_json_dict(self.config.rqvae),
                },
            )
        return self

    @torch.no_grad()
    def encode(self, embeddings: torch.Tensor) -> torch.LongTensor:
        x = self.normalizer.transform(embeddings.float())
        device = torch.device(
            self.config.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(device)
        return self.model.encode(x, batch_size=self.config.encode_batch_size)

    @torch.no_grad()
    def encode_store(self, item_ids: Sequence[str], embeddings: torch.Tensor) -> SemanticIDStore:
        codes = self._prepare_export_codes(self.encode(embeddings))
        resolver = SemanticIDCollisionResolver(
            self.config.collision_policy,
            suffix_offset=self._collision_suffix_offset(),
            suffix_size=self._collision_suffix_size(),
            seed=self.config.collision_seed,
        )
        resolved, stats = resolver.resolve(item_ids, codes)
        self.collision_stats = stats
        store = SemanticIDStore(
            codes_to_mapping(item_ids, resolved),
            metadata={
                "tokenizer": "rqvae",
                "normalize": self.config.normalize,
                "collision_stats": stats.to_dict(),
                "codebook_init": self.config.train.codebook_init,
                "level_offset_tokens": self.config.level_offset_tokens,
                "collision_token_offset": self._collision_suffix_offset(),
                "rqvae_config": dataclass_to_json_dict(self.config.rqvae),
            },
        )
        self.semantic_id_store = store
        return store

    def _prepare_export_codes(self, codes: torch.LongTensor) -> torch.LongTensor:
        if self.config.level_offset_tokens:
            return offset_code_tokens(codes, codebook_size=self.config.rqvae.codebook_size)
        return codes.long()

    def _collision_suffix_offset(self) -> int:
        if self.config.collision_token_offset is not None:
            return int(self.config.collision_token_offset)
        if self.config.level_offset_tokens:
            return int(self.config.rqvae.num_codebooks * self.config.rqvae.codebook_size)
        return 0

    def _collision_suffix_size(self) -> int | None:
        if self.config.collision_suffix_size is not None:
            return int(self.config.collision_suffix_size)
        if self.config.collision_policy == "random_append":
            return int(self.config.rqvae.codebook_size)
        return None

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_rqvae_checkpoint(
            output_dir / "model.pt",
            self.model,
            self.config.rqvae,
            self.config.train,
            extra={
                "normalize": self.config.normalize,
                "collision_policy": self.config.collision_policy,
                "level_offset_tokens": self.config.level_offset_tokens,
                "collision_token_offset": self.config.collision_token_offset,
                "collision_suffix_size": self.config.collision_suffix_size,
                "collision_seed": self.config.collision_seed,
                "encode_batch_size": self.config.encode_batch_size,
            },
        )
        torch.save(self.normalizer.state_dict(), output_dir / "normalizer.pt")

        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(_config_to_dict(self.config), f, indent=2)

        if self.train_state is not None:
            self.train_state.save_jsonl(output_dir / "training_log.jsonl")
            with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_val_loss": self.train_state.best_val_loss,
                        "best_epoch": self.train_state.best_epoch,
                    },
                    f,
                    indent=2,
                )

        if self.semantic_id_store is not None:
            self.semantic_id_store.save_json(output_dir / "semantic_ids.json")
            self.semantic_id_store.save_pt(output_dir / "semantic_ids.pt")
            with open(output_dir / "semantic_id_metadata.json", "w", encoding="utf-8") as f:
                json.dump(self.semantic_id_store.metadata, f, indent=2)

    @classmethod
    def load(
        cls,
        output_dir: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "RQVAEItemTokenizer":
        output_dir = Path(output_dir)
        with open(output_dir / "config.json", "r", encoding="utf-8") as f:
            cfg_raw = json.load(f)
        cfg = _config_from_dict(cfg_raw)
        tokenizer = cls(cfg)
        model, _, _ = load_rqvae_checkpoint(output_dir / "model.pt", map_location=map_location)
        tokenizer.model = model
        norm_state = torch.load(output_dir / "normalizer.pt", map_location="cpu")
        tokenizer.normalizer.load_state_dict(norm_state)
        sid_path = output_dir / "semantic_ids.json"
        if sid_path.exists():
            tokenizer.semantic_id_store = SemanticIDStore.load_json(sid_path)
        return tokenizer


def _config_to_dict(config: RQVAEItemTokenizerConfig) -> dict[str, Any]:
    return {
        "rqvae": asdict(config.rqvae),
        "train": asdict(config.train),
        "normalize": config.normalize,
        "collision_policy": config.collision_policy,
        "level_offset_tokens": config.level_offset_tokens,
        "collision_token_offset": config.collision_token_offset,
        "collision_suffix_size": config.collision_suffix_size,
        "collision_seed": config.collision_seed,
        "encode_batch_size": config.encode_batch_size,
    }


def _config_from_dict(data: dict[str, Any]) -> RQVAEItemTokenizerConfig:
    rqvae_cfg = RQVAEConfig(**data["rqvae"])
    train_cfg = RQVAETrainConfig(**_normalize_train_config_dict(data.get("train", {})))
    return RQVAEItemTokenizerConfig(
        rqvae=rqvae_cfg,
        train=train_cfg,
        normalize=data.get("normalize", "l2"),
        collision_policy=data.get("collision_policy", "append"),
        level_offset_tokens=bool(data.get("level_offset_tokens", True)),
        collision_token_offset=data.get("collision_token_offset"),
        collision_suffix_size=data.get("collision_suffix_size"),
        collision_seed=int(data.get("collision_seed", 42)),
        encode_batch_size=int(data.get("encode_batch_size", 4096)),
    )
