from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class ScalableCrossEntropyConfig:
    """Configuration for Scalable Cross-Entropy (SCE).

    This is an adaptation of the MIPS/bucketing objective from
    AIRI-Institute/Scalable-SASRec (RecSys'24).  The algorithm approximates the
    full item softmax by selecting informative (hidden-state, item) pairs in a
    small number of random-projection buckets.

    ``bucket_size_x`` controls how many supervised hidden states each bucket
    keeps; ``bucket_size_y`` controls how many item classes it keeps.
    """

    n_buckets: int = 160
    bucket_size_x: int = 160
    bucket_size_y: int = 1024
    mix_x: bool = False

    def __post_init__(self) -> None:
        for name in ("n_buckets", "bucket_size_x", "bucket_size_y"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{name} must be a positive integer; got {value!r}. "
                    "('auto' bucket sizes are resolved from the dataset's "
                    "batch_size and window by the exp1 runner before the loss "
                    "is built.)"
                )
        if self.n_buckets <= 0:
            raise ValueError("n_buckets must be positive.")
        if self.bucket_size_x <= 0:
            raise ValueError("bucket_size_x must be positive.")
        if self.bucket_size_y <= 0:
            raise ValueError("bucket_size_y must be positive.")

    @classmethod
    def from_dict(
        cls,
        data: dict | None,
    ) -> "ScalableCrossEntropyConfig":
        return cls(**(dict(data) if data is not None else {}))

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "n_buckets": int(self.n_buckets),
            "bucket_size_x": int(self.bucket_size_x),
            "bucket_size_y": int(self.bucket_size_y),
            "mix_x": bool(self.mix_x),
        }


class ScalableCrossEntropyLoss(nn.Module):
    """Approximate full item cross-entropy without materializing [N, V] logits.

    Args to ``forward``:
      - hidden_states: [B, T, D] or [N, D] predictor representations;
      - target_item_ids: [B, T] or [N] positive item IDs;
      - item_embedding_weight: [V, D] output/item embedding matrix;
      - target_mask: optional mask aligned with targets.

    Padding targets are removed before bucketing.  The padding item is also not
    considered as a negative class, matching the reference SCE implementation.
    The exact ``item_loss: cross_entropy`` objective keeps the padding item as a
    zero-logit competitor in its softmax, so the two losses differ by a small
    constant offset; that offset is expected.  When every bucket holds every
    supervised position and every non-padding item, this loss equals full
    cross-entropy with the padding column removed
    (``tests/test_scalable_cross_entropy.py``).

    SCE is a stochastic training-time approximation: ``SFTMethod`` only uses it
    for ``stage == "train"`` and keeps the exact loss for validation/test.
    """

    def __init__(
        self,
        cfg: ScalableCrossEntropyConfig | None = None,
        *,
        pad_item_id: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ScalableCrossEntropyConfig()
        self.pad_item_id = int(pad_item_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_item_ids: torch.Tensor,
        item_embedding_weight: torch.Tensor,
        *,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim not in (2, 3):
            raise ValueError(
                "hidden_states must have shape [N, D] or [B, T, D]. "
                f"Got {tuple(hidden_states.shape)}."
            )
        if item_embedding_weight.ndim != 2:
            raise ValueError("item_embedding_weight must have shape [V, D].")
        if hidden_states.size(-1) != item_embedding_weight.size(-1):
            raise ValueError(
                "Hidden size and item embedding size must match: "
                f"{hidden_states.size(-1)} != {item_embedding_weight.size(-1)}."
            )

        x = hidden_states.reshape(-1, hidden_states.size(-1))
        y = target_item_ids.to(hidden_states.device).long().reshape(-1)
        if y.numel() != x.size(0):
            raise ValueError(
                "target_item_ids must align with hidden_states before flattening: "
                f"hidden={tuple(hidden_states.shape)}, target={tuple(target_item_ids.shape)}."
            )

        valid = y.ne(self.pad_item_id)
        if target_mask is not None:
            mask = target_mask.to(hidden_states.device).bool().reshape(-1)
            if mask.numel() != y.numel():
                raise ValueError("target_mask must have the same number of elements as targets.")
            valid &= mask

        if not bool(valid.any()):
            raise ValueError("Scalable cross-entropy received no valid supervised targets.")

        x = x[valid]
        y = y[valid]
        w = item_embedding_weight

        # One host sync for the whole range check (the boolean indexing above
        # already forced one) instead of separate .item() calls for min and max.
        y_min, y_max = (int(v) for v in torch.stack(torch.aminmax(y)).tolist())
        if y_min < 0 or y_max >= w.size(0):
            raise ValueError(
                "target_item_ids contain values outside the item embedding table: "
                f"min={y_min}, max={y_max}, V={w.size(0)}."
            )

        n_examples, hidden_dim = x.shape
        num_classes = int(w.size(0))
        num_nonpad_classes = num_classes - int(0 <= self.pad_item_id < num_classes)
        if num_nonpad_classes <= 0:
            raise ValueError("SCE requires at least one non-padding item class.")

        bucket_size_x = min(int(self.cfg.bucket_size_x), n_examples)
        bucket_size_y = min(int(self.cfg.bucket_size_y), num_nonpad_classes)
        n_buckets = int(self.cfg.n_buckets)

        correct_logits = (x * w.index_select(0, y)).sum(dim=-1)

        # Bucket assignment is discrete and intentionally excluded from autograd.
        # Gradients still flow through the selected x/w vectors below.
        with torch.no_grad():
            x_detached = x.detach()
            w_detached = w.detach()
            scale = float(hidden_dim) ** -0.25

            if self.cfg.mix_x:
                omega = torch.randn(
                    n_examples,
                    n_buckets,
                    device=x.device,
                    dtype=x.dtype,
                ) * scale
                buckets = omega.transpose(0, 1).matmul(x_detached)
            else:
                buckets = torch.randn(
                    n_buckets,
                    hidden_dim,
                    device=x.device,
                    dtype=x.dtype,
                ) * scale

            x_projection = buckets.matmul(x_detached.transpose(0, 1))
            top_x = torch.topk(x_projection, k=bucket_size_x, dim=1).indices

            item_projection = buckets.matmul(w_detached.transpose(0, 1))
            if 0 <= self.pad_item_id < num_classes:
                item_projection[:, self.pad_item_id] = -torch.inf
            top_y = torch.topk(item_projection, k=bucket_size_y, dim=1).indices

        selected_x = x[top_x]  # [n_buckets, bucket_size_x, D]
        selected_w = w[top_y]  # [n_buckets, bucket_size_y, D]

        wrong_logits = selected_x.matmul(selected_w.transpose(-1, -2))
        selected_targets = y[top_x]
        is_positive_class = selected_targets.unsqueeze(-1).eq(top_y.unsqueeze(1))
        wrong_logits = wrong_logits.masked_fill(is_positive_class, -torch.inf)

        selected_correct_logits = correct_logits[top_x].unsqueeze(-1)
        logits = torch.cat((wrong_logits, selected_correct_logits), dim=-1)
        labels = torch.full(
            logits.shape[:-1],
            logits.size(-1) - 1,
            device=logits.device,
            dtype=torch.long,
        )
        bucket_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).reshape(n_buckets, bucket_size_x)

        # One position can be selected by several buckets.  As in the reference
        # implementation, keep its hardest (maximum) bucket loss and average over
        # positions selected by at least one bucket.  Selection is tracked with
        # an explicit mask rather than by filtering finite values: a diverged
        # model yields NaN bucket losses, and those must surface as a NaN loss
        # (exactly like full cross-entropy) so the trainer's non-finite-loss
        # handling applies, instead of being mistaken for "no position selected".
        flat_top_x = top_x.reshape(-1)
        per_example_loss = torch.full(
            (n_examples,),
            -torch.inf,
            device=x.device,
            dtype=bucket_losses.dtype,
        )
        per_example_loss.scatter_reduce_(
            0,
            flat_top_x,
            bucket_losses.reshape(-1),
            reduce="amax",
            include_self=True,
        )
        selected = torch.zeros(n_examples, dtype=torch.bool, device=x.device)
        selected[flat_top_x] = True
        return per_example_loss[selected].mean()

