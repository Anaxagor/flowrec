"""Gradient-norm calibration of a composite objective's component weight.

Every FlowRec fine-tuning method that mixes an anchor loss with an auxiliary
loss shares the same shape::

    L = w_ref * L_ref + w_comp * L_comp      e.g. L_SFT + lambda * L_GFlow
                                                  L_SFT + lambda * L_GRPO

This module picks ``lambda`` so that, at the calibration point, the auxiliary
gradient has a target norm relative to the reference gradient::

    lambda = rho_target / median_b( ||g_comp,b|| / ||g_ref,b|| )

The maths (:func:`calibrate_from_loss_pairs`) only needs the two *unweighted*
component losses on the same batch, so it is independent of the model, the
dataset and the method. Composite methods expose their unweighted components
under ``MethodOutput.extra[LOSS_COMPONENTS_KEY]`` (GFlow: ``sft``/``gflow``,
MiniOneRec: ``sft``/``grpo``, DPO: ``sft``/``dpo``, SDPO: ``sft``/``sdpo``), which lets
:func:`calibrate_component_weight` calibrate any of them with one forward pass
per batch::

    result = calibrate_component_weight(
        model, method, batches, component="gflow", reference="sft",
        target_ratio=1.0, num_batches=5, seed=1234, device=device,
    )
    method = GFlowMethod(replace(cfg, gflow_weight=result["weight"]), ...)

The exp4 scripts keep their original two-variant protocol (a method instance
per isolated component, identical RNG per pass) through
:func:`variant_loss_pairs` and :func:`calibrate_sequential_variants`. The
sequential helper frees each variant graph before the next forward. It is
not a replacement for :func:`calibrate_component_weight`: SDPO and GRPO
still read both losses from one shared forward.

Gradients are taken with ``torch.autograd.grad``: ``.grad`` buffers, optimizer
state, AMP scaling and gradient clipping are never involved, so the ratio is
the true full-precision one. Model/method train modes and every RNG the
forward may consume are restored afterwards, so calibration never shifts the
training run that follows it.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Iterator, Sequence

import torch

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY, FlowRecMethod, MethodOutput
from flowrec.models.base import Batch, FlowRecModel

__all__ = [
    "LOSS_COMPONENTS_KEY",
    "calibrate_component_weight",
    "calibrate_from_loss_pairs",
    "calibrate_sequential_variants",
    "gradient_vector",
    "loss_components",
    "variant_loss_pairs",
]


# --------------------------------------------------------------------------- #
# Gradient maths
# --------------------------------------------------------------------------- #
def gradient_vector(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Flat gradient of ``loss`` over ``params`` (zeros for unused params).

    Uses ``torch.autograd.grad`` so ``.grad`` buffers and optimizer state stay
    untouched; the result is the true unscaled gradient (no clipping, no AMP
    scaling, no optimizer preconditioning). A loss that carries no graph at
    all (e.g. a constant placeholder) yields an all-zero vector so the caller
    can fail with a meaningful "zero gradient" message.
    """
    params = list(params)
    if not loss.requires_grad:
        return torch.cat([torch.zeros_like(p).float().reshape(-1) for p in params])
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph)
    pieces = [
        (g if g is not None else torch.zeros_like(p)).detach().float().reshape(-1)
        for g, p in zip(grads, params)
    ]
    return torch.cat(pieces)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def calibrate_from_loss_pairs(
    loss_pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    params: Sequence[torch.nn.Parameter],
    *,
    target_ratio: float,
    eps: float = 1e-12,
    reference_name: str = "sft",
    component_name: str = "gflow",
) -> dict[str, Any]:
    """``weight = target_ratio / median_b(||g_comp,b|| / ||g_ref,b||)``.

    ``loss_pairs`` yields ``(loss_ref, loss_comp)``: the *unweighted* reference
    and component losses computed on the same batch. The two may share one
    autograd graph (single forward pass) or come from separate passes.
    Gradient cosine is DIAGNOSTIC ONLY. Raises on degenerate or non-finite
    norms instead of silently clipping.

    The returned dict carries generic keys (``weight``, ``reference``,
    ``component``, ``raw_grad_ratio_median``, ...) plus keys named after the
    components (``fixed_<component>_weight``, ``<reference>_grad_norm_median``,
    ``<component>_grad_norm_median``, ``per_batch/<name>_norms``). With the
    default names those are exactly the keys the exp4 calibration.json files
    use.
    """
    params = list(params)

    def gradient_pairs() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for loss_ref, loss_comp in loss_pairs:
            # Retain the graph for the first gradient: when both losses come
            # from the same forward pass they share saved activations.
            g_ref = gradient_vector(loss_ref, params, retain_graph=True)
            g_comp = gradient_vector(loss_comp, params)
            yield g_ref, g_comp
            del loss_ref, loss_comp

    return _calibrate_from_gradient_pairs(
        gradient_pairs(),
        target_ratio=target_ratio,
        eps=eps,
        reference_name=reference_name,
        component_name=component_name,
    )


def _calibrate_from_gradient_pairs(
    gradient_pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    target_ratio: float,
    eps: float,
    reference_name: str,
    component_name: str,
) -> dict[str, Any]:
    """Shared statistics for joint-graph and sequential calibration."""
    ref_norms: list[float] = []
    comp_norms: list[float] = []
    ratios: list[float] = []
    cosines: list[float] = []
    for g_ref, g_comp in gradient_pairs:
        norm_ref = float(g_ref.norm().item())
        norm_comp = float(g_comp.norm().item())
        cosine = float((g_ref @ g_comp).item() / (norm_ref * norm_comp + eps))
        del g_ref, g_comp
        ref_norms.append(norm_ref)
        comp_norms.append(norm_comp)
        ratios.append(norm_comp / (norm_ref + eps))
        cosines.append(cosine)

    if not ratios:
        raise ValueError("Calibration received no batches.")
    if not all(math.isfinite(v) for v in ref_norms + comp_norms):
        raise ValueError(
            f"Calibration produced non-finite gradient norms: "
            f"{reference_name}={ref_norms}, {component_name}={comp_norms}"
        )

    ref_median = _median(ref_norms)
    comp_median = _median(comp_norms)
    ratio_median = _median(ratios)
    if ref_median <= eps:
        raise ValueError(
            f"Median {reference_name} gradient norm is effectively zero ({ref_median:.3e}); "
            f"the reference loss has no gradient - check the loss / checkpoint."
        )
    if comp_median <= eps:
        raise ValueError(
            f"Median {component_name} gradient norm is effectively zero ({comp_median:.3e}); "
            f"the component loss has no gradient - check its inputs (rewards, candidates, rollouts)."
        )
    weight = float(target_ratio) / ratio_median
    if not math.isfinite(weight):
        raise ValueError(f"Calibrated weight is not finite (ratio_median={ratio_median:.3e}).")
    if ratio_median > 1e4 or ratio_median < 1e-4:
        print(
            f"[calibration] WARNING: extreme raw gradient ratio r0={ratio_median:.3e} "
            f"-> {component_name} weight={weight:.3e}; not clipping.",
            flush=True,
        )

    mean_ratio = sum(ratios) / len(ratios)
    std_ratio = (sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios)) ** 0.5
    return {
        "reference": reference_name,
        "component": component_name,
        "target_grad_ratio": float(target_ratio),
        "weight": weight,
        f"fixed_{component_name}_weight": weight,
        f"{reference_name}_grad_norm_median": ref_median,
        f"{component_name}_grad_norm_median": comp_median,
        "raw_grad_ratio_median": ratio_median,
        "raw_grad_ratio_mean": mean_ratio,
        "raw_grad_ratio_std": std_ratio,
        "grad_cosine_median": _median(cosines),
        "effective_ratio_check": weight * ratio_median,
        "num_batches": len(ratios),
        "per_batch": {
            f"{reference_name}_norms": ref_norms,
            f"{component_name}_norms": comp_norms,
            "ratios": ratios,
            "cosines": cosines,
        },
    }


# --------------------------------------------------------------------------- #
# Method-level drivers
# --------------------------------------------------------------------------- #
def loss_components(output: MethodOutput, *, method_name: str | None = None) -> dict[str, torch.Tensor]:
    """The unweighted loss components a composite method reported.

    Raises when the method does not expose them; every composite FlowRec
    method sets ``extra[LOSS_COMPONENTS_KEY] = {name: unweighted_loss}``.
    """
    components = output.extra.get(LOSS_COMPONENTS_KEY)
    if not isinstance(components, dict) or not components:
        who = f"Method {method_name!r}" if method_name else "The method"
        raise ValueError(
            f"{who} does not expose loss components: compute_loss must set "
            f"extra[{LOSS_COMPONENTS_KEY!r}] = {{name: unweighted loss tensor}}."
        )
    return {str(k): v for k, v in components.items()}


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_snapshot() -> dict[str, Any]:
    return {
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": _numpy_state(),
    }


def _numpy_state() -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a hard dependency in practice
        return None
    return np.random.get_state()


def _rng_restore(snapshot: dict[str, Any]) -> None:
    torch.random.set_rng_state(snapshot["torch"])
    if snapshot["cuda"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda"])
    random.setstate(snapshot["python"])
    if snapshot["numpy"] is not None:
        import numpy as np

        np.random.set_state(snapshot["numpy"])


def _batches(
    batches: Iterable[Batch],
    *,
    num_batches: int | None,
    device: torch.device | str | None,
) -> Iterator[Batch]:
    if num_batches is not None and num_batches < 1:
        raise ValueError("num_batches must be >= 1 when given.")
    target = torch.device(device) if device is not None else None
    if target is not None:
        from flowrec.training.trainer import move_to_device  # local import: trainer imports ft_methods

    count = 0
    for batch in batches:
        if num_batches is not None and count >= num_batches:
            break
        if target is not None:
            batch = move_to_device(batch, target)
        yield batch
        count += 1
    if num_batches is not None and count < num_batches:
        raise ValueError(
            f"Calibration asked for {num_batches} batches but the iterable yielded only {count}."
        )


def variant_loss_pairs(
    model: FlowRecModel,
    reference_method: FlowRecMethod,
    component_method: FlowRecMethod,
    batches: Iterable[Batch],
    *,
    seed: int | None = None,
    step: int | None = 0,
    stage: str = "train",
    num_batches: int | None = None,
    device: torch.device | str | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Two-variant protocol: ``(reference_method.loss, component_method.loss)`` per batch.

    ``reference_method`` / ``component_method`` are method instances whose
    weights isolate one unweighted term each (e.g. GFlow with
    ``(sft_weight, gflow_weight) = (1, 0)`` and ``(0, 1)``). Before each
    variant's forward the torch RNG is re-seeded with ``seed + 1000 * b`` so
    dropout masks and sampled candidates match across the pair. ``step=0``
    keeps curriculum samplers in their initial phase - the gradient a run
    actually starts with. The caller owns model mode and RNG restoration
    (the exp4 scripts call ``seed_everything`` afterwards).
    """
    for index, batch in enumerate(_batches(batches, num_batches=num_batches, device=device)):
        pair: list[torch.Tensor] = []
        for variant in (reference_method, component_method):
            if seed is not None:
                _seed_torch(int(seed) + 1000 * index)
            out = variant.compute_loss(model, batch, stage=stage, step=step)
            pair.append(out.loss)
        yield pair[0], pair[1]


def calibrate_sequential_variants(
    model: FlowRecModel,
    reference_method: FlowRecMethod,
    component_method: FlowRecMethod,
    batches: Iterable[Batch],
    params: Sequence[torch.nn.Parameter],
    *,
    target_ratio: float,
    eps: float = 1e-12,
    seed: int | None = None,
    step: int | None = 0,
    stage: str = "train",
    num_batches: int | None = None,
    device: torch.device | str | None = None,
    reference_name: str = "sft",
    component_name: str = "gflow",
) -> dict[str, Any]:
    """Two-variant FP32 calibration with at most one live forward graph.

    The forward RNG is reset exactly as in :func:`variant_loss_pairs`. Each
    variant's gradient is computed and its graph released before the other
    variant runs; only the detached parameter gradient survives. This is the
    protocol for two isolated method instances. SDPO and GRPO keep
    :func:`calibrate_component_weight`, which reads both unweighted losses
    from one shared forward and does not draw a second sample.

    The caller owns train modes and restoring RNG state. Parameter ``.grad``
    buffers and optimizer state are untouched.
    """
    params = list(params)

    def gradient_pairs() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for index, batch in enumerate(_batches(batches, num_batches=num_batches, device=device)):
            gradients: list[torch.Tensor] = []
            for variant in (reference_method, component_method):
                if seed is not None:
                    _seed_torch(int(seed) + 1000 * index)
                output = variant.compute_loss(model, batch, stage=stage, step=step)
                gradients.append(gradient_vector(output.loss, params))
                del output
            yield gradients[0], gradients[1]

    return _calibrate_from_gradient_pairs(
        gradient_pairs(),
        target_ratio=target_ratio,
        eps=eps,
        reference_name=reference_name,
        component_name=component_name,
    )


def calibrate_component_weight(
    model: FlowRecModel,
    method: FlowRecMethod,
    batches: Iterable[Batch],
    *,
    component: str,
    reference: str = "sft",
    target_ratio: float = 1.0,
    num_batches: int | None = None,
    seed: int | None = None,
    step: int | None = 0,
    stage: str = "train",
    device: torch.device | str | None = None,
    params: Sequence[torch.nn.Parameter] | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Calibrate ``component``'s weight against ``reference`` for any composite method.

    One forward pass of ``method`` per batch; the unweighted losses are read
    from ``extra[LOSS_COMPONENTS_KEY]`` (so the method's current weights are
    irrelevant), and both gradients share the pass - identical dropout masks,
    candidates and rollouts by construction. Works for GFlow
    (``component="gflow"``), MiniOneRec/GRPO (``component="grpo"``), DPO
    (``component="dpo"``, needs ``sft_weight > 0`` so SFT is computed), SDPO
    (``component="sft"``, ``reference="sdpo"``) and any
    future method that reports its components.

    ``params`` defaults to every trainable model *and* method parameter, the
    set the :class:`flowrec.training.Trainer` optimizes. ``seed`` re-seeds the
    torch RNG per batch (``seed + 1000 * b``) for reproducible sampling; model
    and method are put in train mode (dropout on, as in training) and every
    RNG state is restored afterwards. No autocast: full precision. The result
    is :func:`calibrate_from_loss_pairs`'s dict; apply ``result["weight"]`` to
    the method config (e.g. ``replace(cfg, gflow_weight=result["weight"])``).
    """
    if component == reference:
        raise ValueError("component and reference must name different loss components.")
    if params is None:
        params = [
            p for p in list(model.parameters()) + list(method.parameters()) if p.requires_grad
        ]
    params = list(params)
    if not params:
        raise ValueError("No trainable parameters to calibrate against.")

    snapshot = _rng_snapshot()
    was_model_training = model.training
    was_method_training = method.training
    model.train(True)
    method.train(True)

    def pairs() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for index, batch in enumerate(_batches(batches, num_batches=num_batches, device=device)):
            if seed is not None:
                _seed_torch(int(seed) + 1000 * index)
            out = method.compute_loss(model, batch, stage=stage, step=step)
            components = loss_components(out, method_name=getattr(method, "method_name", None))
            missing = [name for name in (reference, component) if name not in components]
            if missing:
                raise ValueError(
                    f"Loss component(s) {missing} not reported by method "
                    f"{getattr(method, 'method_name', type(method).__name__)!r}; "
                    f"available: {sorted(components)}."
                )
            yield components[reference], components[component]

    try:
        return calibrate_from_loss_pairs(
            pairs(),
            params,
            target_ratio=float(target_ratio),
            eps=float(eps),
            reference_name=reference,
            component_name=component,
        )
    finally:
        model.train(was_model_training)
        method.train(was_method_training)
        _rng_restore(snapshot)
