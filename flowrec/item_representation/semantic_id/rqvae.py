from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


CodebookInitMode = Literal["data", "kmeans"]


@dataclass(frozen=True)
class RQVAEConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    num_codebooks: int = 3
    codebook_size: int = 256
    commitment_weight: float = 0.25
    dropout: float = 0.0
    use_batch_norm: bool = False


@dataclass(frozen=True)
class RQVAETrainConfig:
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: Literal["adamw", "adam"] = "adamw"
    val_ratio: float = 0.05
    num_workers: int = 0
    drop_last: bool = False
    seed: int = 42
    gradient_clip_norm: float | None = 1.0
    log_every: int = 50
    progress_bar: bool = True
    device: str | None = None
    codebook_init: CodebookInitMode = "data"
    kmeans_init_iters: int = 20
    kmeans_init_on_cpu: bool = True
    kmeans_init_tolerance: float = 1e-6
    dead_code_reset_interval: int | None = 1


@dataclass
class RQVAETrainState:
    train_log: list[dict[str, float | int]] = field(default_factory=list)
    best_val_loss: float | None = None
    best_epoch: int | None = None

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in self.train_log:
                f.write(json.dumps(row) + "\n")


class MLP(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        *,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output dimensions")

        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            in_dim, out_dim = int(dims[idx]), int(dims[idx + 1])
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == len(dims) - 2
            if not is_last or final_activation:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer used by RQ-VAE.

    Given an encoder vector z, the quantizer greedily selects one code from each
    codebook to approximate z with a sum of code vectors.
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        code_dim: int,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if num_codebooks <= 0:
            raise ValueError("num_codebooks must be positive")
        if codebook_size <= 1:
            raise ValueError("codebook_size must be > 1")
        if code_dim <= 0:
            raise ValueError("code_dim must be positive")

        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.commitment_weight = float(commitment_weight)

        self.codebooks = nn.Parameter(
            torch.empty(self.num_codebooks, self.codebook_size, self.code_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Uniform range similar to VQ-VAE initializations.
        bound = 1.0 / max(1, self.codebook_size)
        nn.init.uniform_(self.codebooks, -bound, bound)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.ndim != 2 or z.shape[1] != self.code_dim:
            raise ValueError(
                f"Expected z with shape [batch, {self.code_dim}], got {tuple(z.shape)}"
            )

        residual = z
        quantized_sum = torch.zeros_like(z)
        codes: list[torch.Tensor] = []
        codebook_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        commitment_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        avg_code_norm = torch.zeros((), device=z.device, dtype=z.dtype)

        for level in range(self.num_codebooks):
            codebook = self.codebooks[level]
            distances = _squared_euclidean_distance(residual, codebook)
            indices = torch.argmin(distances, dim=1)
            quantized = F.embedding(indices, codebook)

            codes.append(indices)
            codebook_loss = codebook_loss + F.mse_loss(quantized, residual.detach())
            commitment_loss = commitment_loss + F.mse_loss(residual, quantized.detach())
            avg_code_norm = avg_code_norm + quantized.norm(dim=1).mean()

            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()

        codes_tensor = torch.stack(codes, dim=1).long()
        # Straight-through estimator: decoder sees quantized vector; encoder
        # receives gradient as if identity through z.
        quantized_st = z + (quantized_sum - z).detach()
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        return {
            "quantized": quantized_st,
            "quantized_no_st": quantized_sum,
            "codes": codes_tensor,
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "avg_code_norm": avg_code_norm / self.num_codebooks,
        }

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.LongTensor:
        self.eval()
        residual = z
        codes: list[torch.Tensor] = []
        for level in range(self.num_codebooks):
            codebook = self.codebooks[level]
            distances = _squared_euclidean_distance(residual, codebook)
            indices = torch.argmin(distances, dim=1)
            quantized = F.embedding(indices, codebook)
            codes.append(indices)
            residual = residual - quantized
        return torch.stack(codes, dim=1).long()

    @torch.no_grad()
    def quantize_from_codes(self, codes: torch.LongTensor) -> torch.Tensor:
        if codes.ndim != 2 or codes.shape[1] != self.num_codebooks:
            raise ValueError(
                f"codes must have shape [batch, {self.num_codebooks}], got {codes.shape}"
            )
        result = torch.zeros(
            (codes.shape[0], self.code_dim),
            dtype=self.codebooks.dtype,
            device=self.codebooks.device,
        )
        for level in range(self.num_codebooks):
            result += F.embedding(codes[:, level].to(self.codebooks.device), self.codebooks[level])
        return result


class RQVAE(nn.Module):
    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.config = config
        encoder_dims = [config.input_dim, *config.hidden_dims, config.latent_dim]
        decoder_dims = [config.latent_dim, *reversed(config.hidden_dims), config.input_dim]

        self.encoder = MLP(
            encoder_dims,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            final_activation=False,
        )
        self.quantizer = ResidualVectorQuantizer(
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            code_dim=config.latent_dim,
            commitment_weight=config.commitment_weight,
        )
        self.decoder = MLP(
            decoder_dims,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            final_activation=False,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        q = self.quantizer(z)
        x_hat = self.decoder(q["quantized"])
        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss + q["vq_loss"]
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": q["vq_loss"],
            "codebook_loss": q["codebook_loss"],
            "commitment_loss": q["commitment_loss"],
            "avg_code_norm": q["avg_code_norm"],
            "x_hat": x_hat,
            "z": z,
            "codes": q["codes"],
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, batch_size: int = 4096) -> torch.LongTensor:
        self.eval()
        device = next(self.parameters()).device
        codes: list[torch.Tensor] = []
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            z = self.encoder(xb)
            codes.append(self.quantizer.encode(z).cpu())
        return torch.cat(codes, dim=0).long()

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, *, batch_size: int = 4096) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        outs: list[torch.Tensor] = []
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            outs.append(self.forward(xb)["x_hat"].cpu())
        return torch.cat(outs, dim=0)


class RQVAETrainer:
    def __init__(self, model: RQVAE, config: RQVAETrainConfig) -> None:
        self.model = model
        self.config = config
        self.state = RQVAETrainState()

    def fit(self, embeddings: torch.Tensor) -> RQVAETrainState:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [num_items, embedding_dim]")
        _seed_everything(self.config.seed)

        device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(device)

        dataset = TensorDataset(embeddings.float())
        train_dataset, val_dataset = _split_dataset(
            dataset, self.config.val_ratio, self.config.seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=self.config.drop_last,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False,
        ) if len(val_dataset) > 0 else None

        optimizer_name = self.config.optimizer.lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError("RQVAETrainConfig.optimizer must be 'adamw' or 'adam'")

        if self.config.codebook_init == "data":
            _progress_write("Initializing RQ-VAE codebooks from data...")
            self._init_codebooks_from_data(embeddings.float(), device)
        elif self.config.codebook_init == "kmeans":
            _progress_write("Initializing RQ-VAE codebooks with k-means...")
            self._init_codebooks_with_kmeans(embeddings.float(), device)
        else:
            raise ValueError(
                f"Unknown codebook_init={self.config.codebook_init!r}; "
                "expected 'data' or 'kmeans'"
            )

        best_state: dict[str, torch.Tensor] | None = None
        global_step = 0
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running = _MetricAverager()
            progress = _make_progress_bar(
                train_loader,
                enabled=self.config.progress_bar,
                desc=f"rqvae epoch {epoch}/{self.config.epochs}",
            )
            for (xb,) in progress:
                xb = xb.to(device)
                out = self.model(xb)
                loss = out["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                optimizer.step()

                step_metrics = {
                    "loss": float(out["loss"].detach().cpu()),
                    "recon_loss": float(out["recon_loss"].detach().cpu()),
                    "vq_loss": float(out["vq_loss"].detach().cpu()),
                    "codebook_loss": float(out["codebook_loss"].detach().cpu()),
                    "commitment_loss": float(out["commitment_loss"].detach().cpu()),
                }
                global_step += 1

                if (
                    self.config.dead_code_reset_interval is not None
                    and self.config.dead_code_reset_interval > 0
                    and global_step % self.config.dead_code_reset_interval == 0
                ):
                    step_metrics.update(
                        self._reset_dead_codes(embeddings.float(), device)
                    )

                running.update(step_metrics, n=int(xb.shape[0]))
                display_metrics = _format_progress_metrics(step_metrics)
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(display_metrics)
                elif self.config.log_every > 0 and global_step % self.config.log_every == 0:
                    _progress_write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": global_step,
                                **display_metrics,
                            },
                            sort_keys=True,
                        )
                    )

            train_metrics = running.compute(prefix="train_")
            val_metrics = self._evaluate(val_loader, device) if val_loader is not None else {}
            row: dict[str, float | int] = {"epoch": epoch, "step": global_step}
            row.update(train_metrics)
            row.update(val_metrics)
            self.state.train_log.append(row)
            _progress_write(json.dumps(_format_epoch_row(row), sort_keys=True))

            monitor = float(val_metrics.get("val_loss", train_metrics["train_loss"]))
            if self.state.best_val_loss is None or monitor < self.state.best_val_loss:
                self.state.best_val_loss = monitor
                self.state.best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(device)

        return self.state

    @torch.no_grad()
    def _init_codebooks_from_data(self, embeddings: torch.Tensor, device: torch.device) -> None:
        self.model.eval()
        z = self._encode_all(embeddings, device)
        residual = z

        for level in range(self.model.quantizer.num_codebooks):
            indices = _sample_rows(
                residual.shape[0],
                self.model.quantizer.codebook_size,
                device=device,
            )
            self.model.quantizer.codebooks[level].copy_(residual[indices])

            codebook = self.model.quantizer.codebooks[level]
            assignments = torch.argmin(_squared_euclidean_distance(residual, codebook), dim=1)
            residual = residual - F.embedding(assignments, codebook)

    @torch.no_grad()
    def _init_codebooks_with_kmeans(self, embeddings: torch.Tensor, device: torch.device) -> None:
        self.model.eval()
        z = self._encode_all(embeddings, device)
        residual = z

        for level in range(self.model.quantizer.num_codebooks):
            centroids = _kmeans(
                residual,
                n_clusters=self.model.quantizer.codebook_size,
                num_iters=self.config.kmeans_init_iters,
                tolerance=self.config.kmeans_init_tolerance,
                initialize_on_cpu=self.config.kmeans_init_on_cpu,
            ).to(device=device, dtype=self.model.quantizer.codebooks.dtype)
            self.model.quantizer.codebooks[level].copy_(centroids)

            codebook = self.model.quantizer.codebooks[level]
            assignments = torch.argmin(_squared_euclidean_distance(residual, codebook), dim=1)
            residual = residual - F.embedding(assignments, codebook)

    @torch.no_grad()
    def _reset_dead_codes(
        self,
        embeddings: torch.Tensor,
        device: torch.device,
    ) -> dict[str, float]:
        was_training = self.model.training
        try:
            self.model.eval()
            z = self._encode_all(embeddings, device)
            residual = z
            metrics: dict[str, float] = {}

            for level in range(self.model.quantizer.num_codebooks):
                codebook = self.model.quantizer.codebooks[level]
                assignments = torch.argmin(_squared_euclidean_distance(residual, codebook), dim=1)
                counts = torch.bincount(
                    assignments,
                    minlength=self.model.quantizer.codebook_size,
                )
                dead_mask = counts == 0
                num_dead = int(dead_mask.sum().item())
                metrics[f"dead_codes_l{level}"] = float(num_dead)

                if num_dead > 0:
                    replacement_rows = _sample_rows(residual.shape[0], num_dead, device=device)
                    codebook[dead_mask] = residual[replacement_rows]
                    assignments = torch.argmin(
                        _squared_euclidean_distance(residual, codebook), dim=1
                    )

                residual = residual - F.embedding(assignments, codebook)

            return metrics
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def _encode_all(self, embeddings: torch.Tensor, device: torch.device) -> torch.Tensor:
        self.model.eval()
        chunks: list[torch.Tensor] = []
        batch_size = max(1, int(self.config.batch_size))
        for start in range(0, embeddings.shape[0], batch_size):
            xb = embeddings[start : start + batch_size].to(device)
            chunks.append(self.model.encoder(xb))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader | None, device: torch.device) -> dict[str, float]:
        if loader is None:
            return {}
        self.model.eval()
        avg = _MetricAverager()
        for (xb,) in loader:
            xb = xb.to(device)
            out = self.model(xb)
            avg.update({
                "loss": float(out["loss"].detach().cpu()),
                "recon_loss": float(out["recon_loss"].detach().cpu()),
                "vq_loss": float(out["vq_loss"].detach().cpu()),
            }, n=int(xb.shape[0]))
        return avg.compute(prefix="val_")


class _MetricAverager:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, metrics: dict[str, float], *, n: int) -> None:
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * n
            self.counts[key] = self.counts.get(key, 0) + n

    def compute(self, *, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}{key}": self.totals[key] / max(1, self.counts[key])
            for key in self.totals
        }


def _make_progress_bar(iterable: Iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, leave=True, dynamic_ncols=True)


def _format_progress_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = ("loss", "recon_loss", "vq_loss", "dead_codes_l0", "dead_codes_l1", "dead_codes_l2", "dead_codes_l3")
    return {
        key: round(float(metrics[key]), 6)
        for key in keys
        if key in metrics
    }


def _format_epoch_row(row: dict[str, float | int]) -> dict[str, float | int]:
    keys = (
        "epoch",
        "step",
        "train_loss",
        "train_recon_loss",
        "train_vq_loss",
        "val_loss",
        "val_recon_loss",
        "val_vq_loss",
        "train_dead_codes_l0",
        "train_dead_codes_l1",
        "train_dead_codes_l2",
        "train_dead_codes_l3",
    )
    result: dict[str, float | int] = {}
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, float):
            result[key] = round(value, 6)
        else:
            result[key] = value
    return result


def _progress_write(message: str) -> None:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print(message, flush=True)
    else:
        tqdm.write(message)


def _split_dataset(dataset: TensorDataset, val_ratio: float, seed: int):
    if val_ratio <= 0 or len(dataset) < 2:
        return dataset, TensorDataset(torch.empty((0, dataset.tensors[0].shape[1])))
    val_size = int(round(len(dataset) * val_ratio))
    val_size = min(max(val_size, 1), len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _squared_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # [B, 1] - 2 [B, D] @ [D, K] + [1, K]
    return (
        x.pow(2).sum(dim=1, keepdim=True)
        - 2.0 * x @ y.t()
        + y.pow(2).sum(dim=1).unsqueeze(0)
    )


@torch.no_grad()
def _kmeans(
    buffer: torch.Tensor,
    *,
    n_clusters: int,
    num_iters: int,
    tolerance: float,
    initialize_on_cpu: bool,
) -> torch.Tensor:
    """Initialize centroids with k-means++ and refine them with Lloyd updates."""
    if buffer.ndim != 2:
        raise ValueError(f"buffer must have shape [num_samples, dim], got {buffer.shape}")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if num_iters < 0:
        raise ValueError("num_iters must be non-negative")

    original_device = buffer.device
    if initialize_on_cpu:
        buffer = buffer.cpu()

    centroids = _kmeans_plus_plus(buffer, n_clusters=n_clusters)
    if num_iters == 0:
        return centroids.to(original_device)

    for _ in range(num_iters):
        old_centroids = centroids.clone()
        assignments = torch.argmin(_squared_euclidean_distance(buffer, centroids), dim=1)

        sums = torch.zeros_like(centroids)
        sums.index_add_(0, assignments, buffer)
        counts = torch.bincount(assignments, minlength=n_clusters)
        non_empty = counts > 0

        next_centroids = centroids.clone()
        next_centroids[non_empty] = (
            sums[non_empty] / counts[non_empty].to(buffer.dtype).unsqueeze(1)
        )

        empty = ~non_empty
        if bool(empty.any()):
            replacement_rows = _sample_rows(
                buffer.shape[0],
                int(empty.sum().item()),
                device=buffer.device,
            )
            next_centroids[empty] = buffer[replacement_rows]

        centroids = next_centroids
        if tolerance > 0 and torch.allclose(
            centroids,
            old_centroids,
            atol=tolerance,
            rtol=0.0,
        ):
            break

    return centroids.to(original_device)


@torch.no_grad()
def _kmeans_plus_plus(buffer: torch.Tensor, *, n_clusters: int) -> torch.Tensor:
    n_samples, n_features = buffer.shape
    if n_samples <= 0:
        raise ValueError("Cannot initialize centroids from an empty tensor")

    centroids = torch.empty(
        (n_clusters, n_features),
        dtype=buffer.dtype,
        device=buffer.device,
    )

    first_centroid_idx = torch.randint(0, n_samples, (1,), device=buffer.device)
    centroids[0] = buffer[first_centroid_idx]
    closest_distances = (
        _squared_euclidean_distance(buffer, centroids[:1]).squeeze(1).clamp_min(0)
    )

    for centroid_idx in range(1, n_clusters):
        distance_sum = closest_distances.sum()
        if (
            not bool(torch.isfinite(distance_sum))
            or float(distance_sum.detach().cpu()) <= 0.0
        ):
            replacement_rows = _sample_rows(
                n_samples,
                n_clusters - centroid_idx,
                device=buffer.device,
            )
            centroids[centroid_idx:] = buffer[replacement_rows]
            break

        next_centroid_idx = torch.multinomial(closest_distances, num_samples=1)
        centroids[centroid_idx] = buffer[next_centroid_idx]
        new_distances = _squared_euclidean_distance(
            buffer,
            centroids[centroid_idx : centroid_idx + 1],
        ).squeeze(1).clamp_min(0)
        closest_distances = torch.minimum(closest_distances, new_distances)

    return centroids


def _sample_rows(num_rows: int, sample_size: int, *, device: torch.device) -> torch.Tensor:
    if num_rows <= 0:
        raise ValueError("Cannot sample from an empty tensor")
    if sample_size <= num_rows:
        return torch.randperm(num_rows, device=device)[:sample_size]
    return torch.randint(num_rows, (sample_size,), device=device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_train_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    legacy_init_from_data = data.pop("init_codebooks_from_data", None)
    if "codebook_init" not in data and legacy_init_from_data is not None:
        if not bool(legacy_init_from_data):
            raise ValueError(
                "Legacy init_codebooks_from_data=False has no equivalent in the new "
                "two-mode API. Set codebook_init to either 'data' or 'kmeans'."
            )
        data["codebook_init"] = "data"
    legacy_tol = data.pop("kmeans_init_tol", None)
    if "kmeans_init_tolerance" not in data and legacy_tol is not None:
        data["kmeans_init_tolerance"] = legacy_tol
    return data


def save_rqvae_checkpoint(
    path: str | Path,
    model: RQVAE,
    rqvae_config: RQVAEConfig,
    train_config: RQVAETrainConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "rqvae_config": asdict(rqvae_config),
        "train_config": asdict(train_config),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_rqvae_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[RQVAE, RQVAEConfig, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location)
    rqvae_config = RQVAEConfig(**payload["rqvae_config"])
    train_config_raw = _normalize_train_config_dict(payload.get("train_config", {}))
    # Validate the saved training config for forward/backward compatibility even
    # though the instantiated model only needs rqvae_config.
    _ = RQVAETrainConfig(**train_config_raw)
    model = RQVAE(rqvae_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location if isinstance(map_location, torch.device) else torch.device(map_location))
    return model, rqvae_config, dict(payload.get("extra", {}))
