from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import svds

from flowrec.models.base import Batch, FlowRecModel, RecModelOutput



SVDInputTransform = Literal["identity", "binary", "log1p", "sqrt"]
SVDFitSource = Literal["all_interactions", "target_positions", "observed_prefix"]
SVDUserRepresentation = Literal["stored", "history_mean", "history_foldin", "hybrid"]


@dataclass(frozen=True, slots=True)
class SVDConfig:
    """Sparse truncated-SVD recommendation baseline.

    The model builds a sparse user-item interaction matrix and factorizes it
    with ``scipy.sparse.linalg.svds``.  It can score users in two ways:

    1. ``user_representation='stored'``:
       use the static user factors learned during SVD fitting.

    2. ``user_representation='history_mean'``:
       infer the current user vector from the mean of item factors in
       ``batch['history_item_ids']``.

    3. ``user_representation='history_foldin'``:
       infer the current user vector from ``batch['history_item_ids']`` via a
       ridge least-squares fold-in step:

           p = xV (V^T V + lambda I)^-1

       where ``V`` is the learned item-factor matrix.

    4. ``user_representation='hybrid'``:
       combine static and history-folded user factors:

           alpha * stored + (1 - alpha) * history_foldin

    The history-aware modes are useful when the batch history should affect
    prediction directly, e.g. rolling-history evaluation or cold users.
    """

    num_users: int
    num_items: int
    n_components: int = 64
    pad_item_id: int = 0
    input_transform: SVDInputTransform = "binary"

    # Prediction behavior.
    user_representation: SVDUserRepresentation = "stored"
    foldin_reg: float = 1e-2
    hybrid_alpha: float = 0.5
    fallback_to_stored_for_empty_history: bool = True
    fallback_to_popularity_for_empty_history: bool = True

    mask_pad_item: bool = True
    exclude_history_in_forward: bool = False
    exclude_history_in_generate: bool = True
    dtype: str = "float32"
    topk_k: int = 10

    # scipy.sparse.linalg.svds options.
    svds_tol: float = 0.0
    svds_maxiter: int | None = None
    svds_solver: str = "arpack"
    random_state: int = 42

    # Optional item chunk size for generate().  This avoids materializing a
    # very large [batch_size, num_items] score matrix.
    score_chunk_size: int | None = None

    # Backward-compatible field.  Ignored because this implementation no longer
    # allocates a dense matrix.
    max_dense_bytes: int | None = None



class SVDModel(FlowRecModel):
    """Sparse-SVD recommender baseline using SciPy.

    Fit once from train interactions, then use as a FlowRec item-scoring model:

        model.fit_from_store(dm.store, dm.splits.train)
        output = model(batch)
        output.scores  # [B, num_items]
    """

    model_type = "svd"

    def __init__(self, cfg: SVDConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(cfg, dict):
            cfg = SVDConfig(**cfg)
        _validate_config(cfg)
        self.cfg = cfg

        k = _effective_rank(cfg.num_users, cfg.num_items, cfg.n_components)

        self.register_buffer("user_factors", torch.zeros(cfg.num_users, k, dtype=torch.float32))
        self.register_buffer("item_factors", torch.zeros(cfg.num_items, k, dtype=torch.float32))
        self.register_buffer("foldin_inverse", torch.eye(k, dtype=torch.float32))
        self.register_buffer("item_fallback_scores", torch.zeros(cfg.num_items, dtype=torch.float32))
        self.register_buffer("item_counts", torch.zeros(cfg.num_items, dtype=torch.float32))
        self.register_buffer("singular_values", torch.zeros(k, dtype=torch.float32))
        self.register_buffer("_is_fitted", torch.tensor(False, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_from_matrix(self, matrix: np.ndarray | torch.Tensor | sparse.spmatrix) -> "SVDModel":
        """Fit from a dense or sparse user-item matrix."""
        x = _to_sparse_matrix(matrix, dtype=self.cfg.dtype)
        if x.ndim != 2:
            raise ValueError("matrix must have shape [num_users, num_items]")
        if x.shape != (self.cfg.num_users, self.cfg.num_items):
            raise ValueError(
                "matrix shape must equal "
                f"({self.cfg.num_users}, {self.cfg.num_items}), got {x.shape}"
            )
        x = self._apply_input_transform(x)
        if self.cfg.mask_pad_item and 0 <= self.cfg.pad_item_id < x.shape[1]:
            x = x.tolil(copy=False)
            x[:, self.cfg.pad_item_id] = 0.0
            x = x.tocsr()
            x.eliminate_zeros()
        counts = np.asarray(x.sum(axis=0)).ravel().astype(np.float32, copy=False)
        self._fit_svd(x, counts=counts)
        return self

    def fit_from_interactions(
        self,
        user_ids: np.ndarray | torch.Tensor | Iterable[int],
        item_ids: np.ndarray | torch.Tensor | Iterable[int],
        values: np.ndarray | torch.Tensor | Iterable[float] | None = None,
    ) -> "SVDModel":
        """Build a sparse matrix from user/item/value triples and fit SVD."""
        user_np = _to_numpy_int(user_ids)
        item_np = _to_numpy_int(item_ids)
        if user_np.shape != item_np.shape:
            raise ValueError("user_ids and item_ids must have the same shape")

        if values is None:
            values_np = np.ones_like(user_np, dtype=np.float32)
        else:
            values_np = _to_numpy_float(values, dtype="float32")
        if values_np.shape != user_np.shape:
            raise ValueError("values must have the same shape as user_ids/item_ids")

        valid = self._valid_user_item_mask(user_np, item_np)
        user_np = user_np[valid]
        item_np = item_np[valid]
        values_np = values_np[valid]

        x = sparse.coo_matrix(
            (values_np.astype(self.cfg.dtype, copy=False), (user_np, item_np)),
            shape=(self.cfg.num_users, self.cfg.num_items),
            dtype=np.dtype(self.cfg.dtype),
        ).tocsr()
        x.sum_duplicates()
        return self.fit_from_matrix(x)

    def fit_from_store(
        self,
        store: Any,
        target_index: Any | None = None,
        *,
        fit_source: SVDFitSource = "observed_prefix",
        use_ratings: bool = False,
    ) -> "SVDModel":
        """Fit from a FlowRec SequenceStore and optional train TargetIndex.

        Args:
            store:
                SequenceStore-like object with ``user_offsets``, ``item_ids`` and
                optional ``ratings``.
            target_index:
                TargetIndex-like object for the train split.
            fit_source:
                - ``all_interactions``: use every interaction in store.
                - ``target_positions``: use only train target positions.
                - ``observed_prefix``: for each train user, use interactions up
                  to the latest train target position, inclusive.
            use_ratings:
                If true, use store.ratings as matrix values; otherwise use 1.0
                per observed interaction.
        """
        user_offsets = np.asarray(store.user_offsets, dtype=np.int64)
        item_ids_all = np.asarray(store.item_ids, dtype=np.int64)
        ratings_all = np.asarray(
            getattr(store, "ratings", np.ones_like(item_ids_all, dtype=np.float32)),
            dtype=np.float32,
        )

        if fit_source == "all_interactions" or target_index is None:
            user_ids = np.repeat(np.arange(len(user_offsets) - 1, dtype=np.int64), np.diff(user_offsets))
            item_ids = item_ids_all
            values = ratings_all if use_ratings else np.ones_like(item_ids, dtype=np.float32)
            return self.fit_from_interactions(user_ids, item_ids, values)

        target_users = np.asarray(target_index.user_indices, dtype=np.int64)
        target_pos_raw = np.asarray(target_index.target_positions, dtype=np.int64)
        target_abs = _target_positions_to_abs(
            user_offsets=user_offsets,
            user_indices=target_users,
            target_positions=target_pos_raw,
            num_interactions=item_ids_all.shape[0],
        )

        if fit_source == "target_positions":
            item_ids = item_ids_all[target_abs]
            values = ratings_all[target_abs] if use_ratings else np.ones_like(item_ids, dtype=np.float32)
            return self.fit_from_interactions(target_users, item_ids, values)

        if fit_source != "observed_prefix":
            raise ValueError(f"Unknown fit_source: {fit_source}")

        # Include interactions from each represented user's sequence start
        # through their latest train target position, inclusive.
        max_end_abs_by_user: dict[int, int] = {}
        for u, abs_pos in zip(target_users.tolist(), target_abs.tolist()):
            u = int(u)
            end_abs = int(abs_pos) + 1
            prev = max_end_abs_by_user.get(u)
            if prev is None or end_abs > prev:
                max_end_abs_by_user[u] = end_abs

        user_chunks: list[np.ndarray] = []
        item_chunks: list[np.ndarray] = []
        value_chunks: list[np.ndarray] = []

        for u, end_abs in max_end_abs_by_user.items():
            if u < 0 or u >= self.cfg.num_users:
                continue
            start_abs = int(user_offsets[u])
            user_end_abs = int(user_offsets[u + 1])
            end_abs = min(int(end_abs), user_end_abs)
            if end_abs <= start_abs:
                continue
            items = item_ids_all[start_abs:end_abs]
            vals = ratings_all[start_abs:end_abs] if use_ratings else np.ones_like(items, dtype=np.float32)
            user_chunks.append(np.full(items.shape[0], u, dtype=np.int64))
            item_chunks.append(items.astype(np.int64, copy=False))
            value_chunks.append(vals.astype(np.float32, copy=False))

        if not user_chunks:
            raise ValueError("No valid train interactions found for SVD fitting.")

        users = np.concatenate(user_chunks)
        items = np.concatenate(item_chunks)
        values = np.concatenate(value_chunks)
        return self.fit_from_interactions(users, items, values)

    def fit_from_dataloader(
        self,
        dataloader: Any,
        *,
        count_targets: bool = True,
        count_history: bool = True,
        use_ratings: bool = False,
    ) -> "SVDModel":
        """Fit from FlowRec dataloader batches.

        Prefer ``fit_from_store`` for indexed datasets.  This fallback may
        over-count histories when examples have overlapping prefixes.
        """
        user_chunks: list[np.ndarray] = []
        item_chunks: list[np.ndarray] = []
        value_chunks: list[np.ndarray] = []

        for batch in dataloader:
            user = _get_first_tensor(batch, ("user_ids", "user_id"))
            if user is None:
                raise KeyError("Batch must contain user_ids or user_id")
            user_np = user.detach().cpu().long().numpy()

            if count_targets:
                target = _get_first_tensor(batch, ("target_item_ids", "target_item_id"))
                if target is not None:
                    target_np = target.detach().cpu().long().numpy()
                    if use_ratings:
                        val = _get_first_tensor(batch, ("target_ratings", "target_rating"))
                        vals = val.detach().cpu().float().numpy() if val is not None else np.ones_like(target_np, dtype=np.float32)
                    else:
                        vals = np.ones_like(target_np, dtype=np.float32)
                    user_chunks.append(user_np)
                    item_chunks.append(target_np)
                    value_chunks.append(vals.astype(np.float32, copy=False))

            if count_history:
                history = _get_first_tensor(batch, ("history_item_ids",))
                if history is not None:
                    history_np = history.detach().cpu().long().numpy()
                    mask = _get_first_tensor(batch, ("history_mask",))
                    if mask is not None:
                        mask_np = mask.detach().cpu().bool().numpy()
                    else:
                        mask_np = history_np != self.cfg.pad_item_id
                    users_rep = np.repeat(user_np, history_np.shape[1])
                    items_flat = history_np.reshape(-1)
                    mask_flat = mask_np.reshape(-1)
                    vals = np.ones_like(items_flat, dtype=np.float32)
                    user_chunks.append(users_rep[mask_flat])
                    item_chunks.append(items_flat[mask_flat])
                    value_chunks.append(vals[mask_flat])

        if not user_chunks:
            raise ValueError("No interactions found for SVD fitting.")

        users = np.concatenate(user_chunks).astype(np.int64, copy=False)
        items = np.concatenate(item_chunks).astype(np.int64, copy=False)
        values = np.concatenate(value_chunks).astype(np.float32, copy=False)
        return self.fit_from_interactions(users, items, values)

    # ------------------------------------------------------------------
    # SVD internals
    # ------------------------------------------------------------------

    def _valid_user_item_mask(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        valid = (users >= 0) & (users < self.cfg.num_users) & (items >= 0) & (items < self.cfg.num_items)
        if self.cfg.mask_pad_item:
            valid &= items != self.cfg.pad_item_id
        return valid

    def _apply_input_transform(self, x: sparse.spmatrix) -> sparse.csr_matrix:
        x = x.tocsr(copy=True)
        x.sum_duplicates()
        if self.cfg.input_transform == "identity":
            pass
        elif self.cfg.input_transform == "binary":
            x.data = np.ones_like(x.data, dtype=x.dtype)
        elif self.cfg.input_transform == "log1p":
            x.data = np.log1p(np.maximum(x.data, 0)).astype(x.dtype, copy=False)
        elif self.cfg.input_transform == "sqrt":
            x.data = np.sqrt(np.maximum(x.data, 0)).astype(x.dtype, copy=False)
        else:
            raise ValueError(f"Unknown input_transform: {self.cfg.input_transform}")
        x.eliminate_zeros()
        return x

    def _fit_svd(self, x: sparse.spmatrix, counts: np.ndarray | None = None) -> None:
        x = x.astype(np.dtype(self.cfg.dtype), copy=False).tocsr()
        min_dim = min(x.shape)
        if min_dim <= 1:
            raise ValueError("SVD requires at least two users and two items.")

        k = min(self.cfg.n_components, min_dim - 1, self.user_factors.size(1))
        if k <= 0:
            raise ValueError("Effective n_components for svds must be positive and < min(matrix.shape).")

        if x.nnz == 0:
            raise ValueError("Cannot fit SVD on an empty interaction matrix.")

        # Sparse truncated SVD. svds usually returns singular values in
        # ascending order, but sort descending explicitly.
        u, s, vt = svds(
            x,
            k=k,
            tol=float(self.cfg.svds_tol),
            maxiter=self.cfg.svds_maxiter,
            which="LM",
            solver=self.cfg.svds_solver,
            random_state=int(self.cfg.random_state),
            return_singular_vectors=True,
        )
        order = np.argsort(s)[::-1]
        s = s[order]
        u = u[:, order]
        vt = vt[order, :]

        user_factors = (u * s[None, :]).astype(np.float32, copy=False)
        item_factors = vt.T.astype(np.float32, copy=False)

        self.user_factors.zero_()
        self.item_factors.zero_()
        self.singular_values.zero_()
        self.user_factors[:, :k].copy_(torch.from_numpy(user_factors))
        self.item_factors[:, :k].copy_(torch.from_numpy(item_factors))
        self.singular_values[:k].copy_(torch.from_numpy(s.astype(np.float32, copy=False)))

        self._recompute_foldin_inverse()

        if counts is None:
            counts = np.asarray(x.sum(axis=0)).ravel().astype(np.float32, copy=False)
        self._set_counts(counts)
        self._is_fitted.fill_(True)

    def _recompute_foldin_inverse(self) -> None:
        """Precompute (V^T V + lambda I)^-1 for history fold-in."""
        item_f = self.item_factors.float()
        k = item_f.size(1)
        gram = item_f.T @ item_f
        reg = float(self.cfg.foldin_reg)
        gram = gram + reg * torch.eye(k, dtype=gram.dtype, device=gram.device)
        inv = torch.linalg.pinv(gram)
        self.foldin_inverse.copy_(inv.to(self.foldin_inverse.device))

    def _set_counts(self, counts: np.ndarray | torch.Tensor) -> None:
        counts_t = torch.as_tensor(counts, dtype=torch.float32)
        if counts_t.numel() != self.cfg.num_items:
            padded = torch.zeros(self.cfg.num_items, dtype=torch.float32)
            n = min(counts_t.numel(), self.cfg.num_items)
            padded[:n] = counts_t[:n]
            counts_t = padded
        self.item_counts.copy_(counts_t.to(self.item_counts.device))

        fallback = torch.log1p(self.item_counts)
        if self.cfg.mask_pad_item:
            fallback[self.cfg.pad_item_id] = -torch.inf
        self.item_fallback_scores.copy_(fallback.to(self.item_fallback_scores.device))

    # ------------------------------------------------------------------
    # FlowRecModel API
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not bool(self._is_fitted.item()):
            raise RuntimeError(
                "SVDModel is not fitted. Call fit_from_matrix(), fit_from_store(), "
                "fit_from_interactions(), or fit_from_dataloader() before scoring."
            )

    def forward(self, batch: Batch) -> RecModelOutput:
        scores = self.score_items(batch)
        return RecModelOutput(logits=scores, scores=scores)

    def score_items(
        self,
        batch: Batch,
        candidate_item_ids: torch.Tensor | None = None,
        candidate_token_ids: torch.Tensor | None = None,
        candidate_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_token_ids is not None:
            raise NotImplementedError("SVDModel scores atomic item IDs, not semantic tokens.")
        self._check_fitted()

        user_ids = _get_first_tensor(batch, ("user_ids", "user_id"))
        if user_ids is None:
            raise KeyError("Batch must contain user_ids or user_id")
        user_ids = user_ids.to(self.device).long().reshape(-1)

        user_factors, fallback_rows = self._get_batch_user_factors(batch, user_ids=user_ids)

        if candidate_item_ids is None:
            scores = user_factors @ self.item_factors.T
            if fallback_rows.any():
                scores = scores.clone()
                scores[fallback_rows] = self.item_fallback_scores
            scores = self._apply_common_masks(scores, batch, exclude_history=self.cfg.exclude_history_in_forward)
            if self.cfg.mask_pad_item:
                scores = scores.clone()
                scores[:, self.cfg.pad_item_id] = -torch.inf
            return scores

        candidate_item_ids = candidate_item_ids.to(user_factors.device).long()
        scores = self._score_candidate_items(
            user_factors=user_factors,
            fallback_rows=fallback_rows,
            candidate_item_ids=candidate_item_ids,
        )
        scores = self._mask_candidate_history_scores(scores, candidate_item_ids, batch, exclude_history=self.cfg.exclude_history_in_forward)
        return scores

    @torch.no_grad()
    def generate(
        self,
        batch: Batch,
        *,
        num_beams: int | None = None,
        num_return_sequences: int | None = None,
        max_new_tokens: int | None = None,
        constrained: bool | None = None,
    ) -> torch.Tensor:
        self._check_fitted()
        k = int(num_return_sequences or self.cfg.topk_k)
        user_ids = _get_first_tensor(batch, ("user_ids", "user_id"))
        if user_ids is None:
            raise KeyError("Batch must contain user_ids or user_id")
        user_ids = user_ids.to(self.device).long().reshape(-1)

        user_factors, fallback_rows = self._get_batch_user_factors(batch, user_ids=user_ids)

        if self.cfg.score_chunk_size is not None and self.cfg.score_chunk_size > 0:
            return self._generate_chunked(batch, user_factors=user_factors, fallback_rows=fallback_rows, k=k)

        scores = user_factors @ self.item_factors.T
        if fallback_rows.any():
            scores = scores.clone()
            scores[fallback_rows] = self.item_fallback_scores
        scores = self._apply_common_masks(scores, batch, exclude_history=self.cfg.exclude_history_in_generate)
        if self.cfg.mask_pad_item:
            scores = scores.clone()
            scores[:, self.cfg.pad_item_id] = -torch.inf
        k = min(k, scores.size(1))
        return torch.topk(scores, k=k, dim=1).indices

    # ------------------------------------------------------------------
    # History-aware user representations
    # ------------------------------------------------------------------

    def _get_batch_user_factors(
        self,
        batch: Batch,
        *,
        user_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.cfg.user_representation
        if mode == "stored":
            return self._stored_user_factors(user_ids)

        if mode == "history_mean":
            history_f, has_history = self._history_mean_user_factors(batch)
            return self._apply_empty_history_fallback(history_f, has_history, user_ids)

        if mode == "history_foldin":
            history_f, has_history = self._history_foldin_user_factors(batch)
            return self._apply_empty_history_fallback(history_f, has_history, user_ids)

        if mode == "hybrid":
            stored_f, stored_fallback = self._stored_user_factors(user_ids)
            history_f, has_history = self._history_foldin_user_factors(batch)
            alpha = float(self.cfg.hybrid_alpha)
            hybrid = alpha * stored_f + (1.0 - alpha) * history_f

            # If a user has no history, use stored factors if possible.
            hybrid = torch.where(has_history[:, None], hybrid, stored_f)
            fallback = (~has_history & stored_fallback) if self.cfg.fallback_to_popularity_for_empty_history else torch.zeros_like(stored_fallback)
            return hybrid, fallback

        raise ValueError(f"Unknown user_representation: {mode}")

    def _stored_user_factors(self, user_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (user_ids >= 0) & (user_ids < self.cfg.num_users)
        safe_user_ids = user_ids.clamp_min(0).clamp_max(self.cfg.num_users - 1)
        factors = self.user_factors[safe_user_ids]
        fallback_rows = ~valid
        return factors, fallback_rows

    def _history_mean_user_factors(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        history, mask = self._history_items_and_mask(batch)
        safe_history = history.clamp_min(0).clamp_max(self.cfg.num_items - 1)
        hist_factors = self.item_factors[safe_history]
        weights = mask.to(hist_factors.dtype).unsqueeze(-1)
        summed = (hist_factors * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        factors = summed / denom
        has_history = mask.any(dim=1)
        return factors, has_history

    def _history_foldin_user_factors(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        history, mask = self._history_items_and_mask(batch)
        safe_history = history.clamp_min(0).clamp_max(self.cfg.num_items - 1)
        hist_factors = self.item_factors[safe_history]
        weights = mask.to(hist_factors.dtype).unsqueeze(-1)
        x_v = (hist_factors * weights).sum(dim=1)  # x @ V
        factors = x_v @ self.foldin_inverse
        has_history = mask.any(dim=1)
        return factors, has_history

    def _apply_empty_history_fallback(
        self,
        history_factors: torch.Tensor,
        has_history: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fallback_rows = ~has_history
        if self.cfg.fallback_to_stored_for_empty_history and fallback_rows.any():
            stored_f, stored_fallback = self._stored_user_factors(user_ids)
            use_stored = fallback_rows & ~stored_fallback
            if use_stored.any():
                history_factors = history_factors.clone()
                history_factors[use_stored] = stored_f[use_stored]
            fallback_rows = fallback_rows & stored_fallback
        if not self.cfg.fallback_to_popularity_for_empty_history:
            fallback_rows = torch.zeros_like(fallback_rows)
        return history_factors, fallback_rows

    def _history_items_and_mask(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        history = _get_first_tensor(batch, ("history_item_ids",))
        if history is None:
            user_ids = _get_first_tensor(batch, ("user_ids", "user_id"))
            if user_ids is None:
                raise KeyError("Batch must contain history_item_ids or user_ids/user_id")
            empty = torch.empty((user_ids.numel(), 0), dtype=torch.long, device=self.device)
            empty_mask = torch.empty((user_ids.numel(), 0), dtype=torch.bool, device=self.device)
            return empty, empty_mask

        history = history.to(self.device).long()
        mask_t = _get_first_tensor(batch, ("history_mask",))
        if mask_t is None:
            mask = history != self.cfg.pad_item_id
        else:
            mask = mask_t.to(self.device).bool()
        mask &= history >= 0
        mask &= history < self.cfg.num_items
        if self.cfg.mask_pad_item:
            mask &= history != self.cfg.pad_item_id
        return history, mask

    # ------------------------------------------------------------------
    # Scoring / generation helpers
    # ------------------------------------------------------------------

    def _score_candidate_items(
        self,
        *,
        user_factors: torch.Tensor,
        fallback_rows: torch.Tensor,
        candidate_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_item_ids.ndim == 1:
            if candidate_item_ids.numel() == user_factors.size(0):
                safe = candidate_item_ids.clamp_min(0).clamp_max(self.cfg.num_items - 1)
                item_f = self.item_factors[safe]
                scores = (user_factors * item_f).sum(dim=-1)
                if fallback_rows.any():
                    scores = scores.clone()
                    scores[fallback_rows] = self.item_fallback_scores[safe[fallback_rows]]
                invalid = (candidate_item_ids < 0) | (candidate_item_ids >= self.cfg.num_items)
                if self.cfg.mask_pad_item:
                    invalid |= candidate_item_ids == self.cfg.pad_item_id
                return scores.masked_fill(invalid, -float("inf"))

            safe = candidate_item_ids.clamp_min(0).clamp_max(self.cfg.num_items - 1)
            item_f = self.item_factors[safe]
            scores = user_factors @ item_f.T
            if fallback_rows.any():
                scores = scores.clone()
                scores[fallback_rows] = self.item_fallback_scores[safe]
            invalid = (candidate_item_ids < 0) | (candidate_item_ids >= self.cfg.num_items)
            if self.cfg.mask_pad_item:
                invalid |= candidate_item_ids == self.cfg.pad_item_id
            if invalid.any():
                scores[:, invalid] = -float("inf")
            return scores

        if candidate_item_ids.ndim == 2:
            if candidate_item_ids.size(0) != user_factors.size(0):
                raise ValueError("candidate_item_ids with shape [B, N] must match batch size")
            safe = candidate_item_ids.clamp_min(0).clamp_max(self.cfg.num_items - 1)
            item_f = self.item_factors[safe]  # [B, N, K]
            scores = (user_factors[:, None, :] * item_f).sum(dim=-1)
            if fallback_rows.any():
                scores = scores.clone()
                scores[fallback_rows] = self.item_fallback_scores[safe[fallback_rows]]
            invalid = (candidate_item_ids < 0) | (candidate_item_ids >= self.cfg.num_items)
            if self.cfg.mask_pad_item:
                invalid |= candidate_item_ids == self.cfg.pad_item_id
            return scores.masked_fill(invalid, -float("inf"))

        raise ValueError("candidate_item_ids must have shape [N], [B], or [B, N]")

    def _generate_chunked(
        self,
        batch: Batch,
        *,
        user_factors: torch.Tensor,
        fallback_rows: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Top-k generation without materializing all item scores at once."""
        k = min(k, self.cfg.num_items)
        chunk = int(self.cfg.score_chunk_size or self.cfg.num_items)
        device = self.device

        top_scores = torch.full((user_factors.size(0), 0), -float("inf"), device=device)
        top_items = torch.empty((user_factors.size(0), 0), dtype=torch.long, device=device)

        seen = None
        if self.cfg.exclude_history_in_generate and "history_item_ids" in batch:
            hist = batch["history_item_ids"].to(device).long()
            mask = batch.get("history_mask")
            if torch.is_tensor(mask):
                mask = mask.to(device).bool()
            else:
                mask = hist != self.cfg.pad_item_id
            seen = hist.masked_fill(~mask, self.cfg.pad_item_id)

        for start in range(0, self.cfg.num_items, chunk):
            end = min(start + chunk, self.cfg.num_items)
            item_f = self.item_factors[start:end]
            scores = user_factors @ item_f.T
            if fallback_rows.any():
                scores = scores.clone()
                scores[fallback_rows] = self.item_fallback_scores[start:end]
            if self.cfg.mask_pad_item and start <= self.cfg.pad_item_id < end:
                scores[:, self.cfg.pad_item_id - start] = -torch.inf
            if seen is not None:
                in_chunk = (seen >= start) & (seen < end)
                if in_chunk.any():
                    local_seen = (seen - start).masked_fill(~in_chunk, 0)
                    scores.scatter_(dim=1, index=local_seen.clamp_min(0).clamp_max(end - start - 1), value=-torch.inf)
            chunk_items = torch.arange(start, end, device=device, dtype=torch.long).expand(scores.size(0), -1)
            merged_scores = torch.cat([top_scores, scores], dim=1)
            merged_items = torch.cat([top_items, chunk_items], dim=1)
            kk = min(k, merged_scores.size(1))
            vals, idx = torch.topk(merged_scores, k=kk, dim=1)
            top_scores = vals
            top_items = merged_items.gather(dim=1, index=idx)

        return top_items

    def _apply_common_masks(self, scores: torch.Tensor, batch: Batch, *, exclude_history: bool) -> torch.Tensor:
        if exclude_history and "history_item_ids" in batch:
            scores = scores.clone()
            history = batch["history_item_ids"].to(scores.device).long()
            mask = batch.get("history_mask")
            if torch.is_tensor(mask):
                mask = mask.to(scores.device).bool()
            else:
                mask = history != self.cfg.pad_item_id
            safe_history = history.masked_fill(~mask, self.cfg.pad_item_id)
            scores.scatter_(
                dim=1,
                index=safe_history.clamp_min(0).clamp_max(scores.size(1) - 1),
                value=-torch.inf,
            )
        return scores

    def _mask_candidate_history_scores(
        self,
        scores: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        batch: Batch,
        *,
        exclude_history: bool,
    ) -> torch.Tensor:
        if not exclude_history or "history_item_ids" not in batch:
            return scores
        history = batch["history_item_ids"].to(scores.device).long()
        mask = batch.get("history_mask")
        if torch.is_tensor(mask):
            mask = mask.to(scores.device).bool()
        else:
            mask = history != self.cfg.pad_item_id
        history = history.masked_fill(~mask, -1)

        if candidate_item_ids.ndim == 1:
            if candidate_item_ids.numel() == scores.numel() and scores.ndim == 1:
                seen = candidate_item_ids[:, None].to(scores.device).eq(history).any(dim=1)
                return scores.masked_fill(seen, -float("inf"))
            if scores.ndim == 2:
                seen = candidate_item_ids.to(scores.device)[None, :].eq(history[:, :, None]).any(dim=1)
                return scores.masked_fill(seen, -float("inf"))
            return scores

        if candidate_item_ids.ndim == 2:
            seen = candidate_item_ids.to(scores.device)[:, None, :].eq(history[:, :, None]).any(dim=1)
            return scores.masked_fill(seen, -float("inf"))

        return scores

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.cfg)


# Backward-compatible aliases.
SVDBaselineConfig = SVDConfig
SVDBaselineModel = SVDModel


def svd_from_dataloader(
    dataloader: Any,
    *,
    num_users: int,
    num_items: int,
    n_components: int = 64,
    pad_item_id: int = 0,
    input_transform: SVDInputTransform = "binary",
    user_representation: SVDUserRepresentation = "stored",
    count_targets: bool = True,
    count_history: bool = True,
) -> SVDModel:
    model = SVDModel(
        SVDConfig(
            num_users=num_users,
            num_items=num_items,
            n_components=n_components,
            pad_item_id=pad_item_id,
            input_transform=input_transform,
            user_representation=user_representation,
        )
    )
    return model.fit_from_dataloader(
        dataloader,
        count_targets=count_targets,
        count_history=count_history,
    )


def _validate_config(cfg: SVDConfig) -> None:
    if cfg.num_users <= 0:
        raise ValueError("num_users must be positive")
    if cfg.num_items <= 1:
        raise ValueError("num_items must include padding plus at least one real item")
    if cfg.n_components <= 0:
        raise ValueError("n_components must be positive")
    if cfg.pad_item_id < 0 or cfg.pad_item_id >= cfg.num_items:
        raise ValueError("pad_item_id must be inside [0, num_items)")
    if cfg.dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if cfg.input_transform not in {"identity", "binary", "log1p", "sqrt"}:
        raise ValueError(f"Unknown input_transform: {cfg.input_transform}")
    if cfg.user_representation not in {"stored", "history_mean", "history_foldin", "hybrid"}:
        raise ValueError(f"Unknown user_representation: {cfg.user_representation}")
    if cfg.foldin_reg <= 0:
        raise ValueError("foldin_reg must be positive")
    if not (0.0 <= cfg.hybrid_alpha <= 1.0):
        raise ValueError("hybrid_alpha must be in [0, 1]")


def _effective_rank(num_users: int, num_items: int, requested: int) -> int:
    min_dim = min(int(num_users), int(num_items))
    if min_dim <= 1:
        return 1
    return max(1, min(int(requested), min_dim - 1))


def _to_sparse_matrix(x: np.ndarray | torch.Tensor | sparse.spmatrix, *, dtype: str) -> sparse.csr_matrix:
    if sparse.issparse(x):
        return x.astype(np.dtype(dtype), copy=False).tocsr()
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy().astype(dtype, copy=False)
    else:
        arr = np.asarray(x, dtype=dtype)
    return sparse.csr_matrix(arr)


def _to_numpy_int(x: np.ndarray | torch.Tensor | Iterable[int]) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.int64, copy=False)
    if isinstance(x, np.ndarray):
        return x.astype(np.int64, copy=False)
    return np.asarray(list(x), dtype=np.int64)


def _to_numpy_float(x: np.ndarray | torch.Tensor | Iterable[float], *, dtype: str) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    return np.asarray(list(x), dtype=dtype)


def _get_first_tensor(batch: Batch, keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _target_positions_to_abs(
    *,
    user_offsets: np.ndarray,
    user_indices: np.ndarray,
    target_positions: np.ndarray,
    num_interactions: int,
) -> np.ndarray:
    """Convert TargetIndex positions to absolute positions if needed.

    New FlowRec indexed TargetIndex stores absolute positions. Older versions
    may store positions local to the user. This helper supports both.
    """
    target_positions = target_positions.astype(np.int64, copy=False)
    user_indices = user_indices.astype(np.int64, copy=False)

    valid_users = (user_indices >= 0) & (user_indices + 1 < len(user_offsets))
    if valid_users.all() and target_positions.size > 0:
        starts = user_offsets[user_indices]
        ends = user_offsets[user_indices + 1]
        looks_absolute = np.all((target_positions >= starts) & (target_positions < ends))
        if looks_absolute:
            return target_positions

    abs_pos = user_offsets[user_indices] + target_positions
    if np.any(abs_pos < 0) or np.any(abs_pos >= num_interactions):
        raise ValueError("target_positions are neither valid absolute nor valid user-local positions")
    return abs_pos.astype(np.int64, copy=False)
