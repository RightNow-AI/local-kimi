"""Session-level prefix lookup, restore, and suffix-only prefill."""

from __future__ import annotations

from typing import Any

import torch

from engine.klinear.generate import prefill
from engine.klinear.state import KLinearDecodeState, MLALayerState

from .key import ModelFingerprint, fingerprint_model, prefix_key, token_tuple
from .store import StateCache


def _model_device(model: object, token_ids: torch.Tensor) -> torch.device:
    if token_ids.device.type != "cpu":
        return token_ids.device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if parameter.device.type != "meta":
                return parameter.device
    return token_ids.device


def _validate_capacity(state: KLinearDecodeState, required_tokens: int) -> None:
    for layer in state.layer_states:
        if isinstance(layer, MLALayerState) and layer.capacity < required_tokens:
            raise ValueError(
                "preallocated restore state is too small for the cached prefix, "
                "suffix, and requested decode capacity"
            )
    if (
        state.attention_mask is not None
        and state.attention_mask.shape[1] < required_tokens
    ):
        raise ValueError("preallocated attention-mask capacity is too small")


@torch.inference_mode()
def warm_prefill(
    model: Any,
    token_ids: torch.Tensor,
    cache: StateCache,
    *,
    into_state: KLinearDecodeState | None = None,
    decode_capacity: int = 1,
    model_fingerprint: ModelFingerprint | None = None,
) -> tuple[KLinearDecodeState, bool]:
    """Return decode-ready state and whether any exact cached prefix was used.

    An exact hit performs no prefill. An extension hit restores the longest
    exact token prefix and advances only the remaining suffix. Before either
    path returns or advances, the compressed snapshot rebuilds MLA projected
    keys and values. The fixed-capacity MLA implementation accepts one token
    at a time, so suffix
    advancement is intentionally incremental. It is exact and preserves the
    restored tensor addresses, but a long suffix does not receive batched
    prefill throughput.

    ``into_state`` should be the state whose buffers a CUDA graph captured. If
    it is omitted, the cache allocates a compatible fixed-capacity state before
    restore. That convenience path is suitable before graph capture and in
    uncaptured tests, but allocation is not part of the hot restore operation.

    The function supports one unpadded sequence. Exact prefix identity is based
    on token ids. Any token change near the start invalidates all longer cached
    prefixes.
    """

    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor")
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
        raise ValueError("token_ids must have shape [1, sequence > 0]")
    if decode_capacity < 0:
        raise ValueError("decode_capacity cannot be negative")

    fingerprint = (
        fingerprint_model(model)
        if model_fingerprint is None
        else model_fingerprint
    )
    tokens = token_tuple(token_ids)
    key = prefix_key(tokens, fingerprint)
    device = _model_device(model, token_ids)

    exact = cache.find_longest_prefix(tokens, fingerprint)
    if exact is not None and exact[1] == len(tokens):
        target = into_state
        if target is None:
            target = cache.allocate_state(
                exact[0],
                model=model,
                device=device,
                additional_tokens=decode_capacity,
            )
        _validate_capacity(target, len(tokens) + decode_capacity)
        if not cache.load(exact[0], target, model=model):
            raise RuntimeError("state-cache entry disappeared during exact restore")
        return target, True

    if exact is not None:
        prefix_cache_key, prefix_count = exact
        additional = len(tokens) - prefix_count + decode_capacity
        target = into_state
        if target is None:
            target = cache.allocate_state(
                prefix_cache_key,
                model=model,
                device=device,
                additional_tokens=additional,
            )
        _validate_capacity(target, len(tokens) + decode_capacity)
        if not cache.load(prefix_cache_key, target, model=model):
            raise RuntimeError("state-cache entry disappeared during prefix restore")

        state = target
        for index in range(prefix_count, len(tokens)):
            output = prefill(model, token_ids[:, index : index + 1], state=state)
            state = output.state
        cache.save(
            key,
            state,
            len(tokens),
            token_ids=tokens,
            model_fingerprint=fingerprint,
        )
        return state, True

    output = prefill(model, token_ids)
    state = output.state.reserve_decode_capacity(decode_capacity)
    cache.save(
        key,
        state,
        len(tokens),
        token_ids=tokens,
        model_fingerprint=fingerprint,
    )
    return state, False
