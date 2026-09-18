"""Preallocated rollback storage for fixed-capacity KLinear decode state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from engine.klinear.state import KDALayerState, KLinearDecodeState, MLALayerState


@dataclass(frozen=True)
class _KDABuffers:
    q_conv: torch.Tensor
    k_conv: torch.Tensor
    v_conv: torch.Tensor
    recurrent: torch.Tensor


@dataclass(frozen=True)
class _MLABuffers:
    offsets: torch.Tensor
    indices: torch.Tensor
    compressed_kv: torch.Tensor
    rotary_key: torch.Tensor
    key_pass: torch.Tensor
    value: torch.Tensor
    position: torch.Tensor


@dataclass(frozen=True)
class _MaskBuffers:
    offsets: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor


class DecodeCheckpoint:
    """Snapshot and restore one fixed-capacity ``KLinearDecodeState``.

    All tensor storage is allocated in the constructor and reused by every
    round. KDA recurrent and convolution tensors are copied in full because a
    decode step mutates them in place. MLA only writes the next fixed ``k``
    cache slots, so the checkpoint copies exactly that window plus every device
    position. This restores every tensor exactly without copying the complete
    context-length MLA cache.

    ``max_speculative_tokens`` is the maximum number of cache slots that any
    round may write. Snapshot and restore use fixed shapes, perform no host
    synchronization, and do not extract device scalars on the host.
    """

    def __init__(
        self,
        state: KLinearDecodeState,
        max_speculative_tokens: int,
    ) -> None:
        if max_speculative_tokens <= 0:
            raise ValueError("max_speculative_tokens must be positive")
        if not state.is_static or state.position is None:
            raise ValueError("DecodeCheckpoint requires fixed-capacity decode state")

        self.max_speculative_tokens = max_speculative_tokens
        self._bound_layers = state.layer_states
        self._bound_attention_mask = state.attention_mask
        self._bound_position = state.position
        self._tokens_seen = state.tokens_seen
        self._position = state.position.clone()

        layer_buffers: list[_KDABuffers | _MLABuffers | None] = []
        for layer_state in state.layer_states:
            if layer_state is None:
                layer_buffers.append(None)
            elif isinstance(layer_state, KDALayerState):
                if not layer_state.is_static:
                    raise ValueError("all KDA layers must use static state")
                layer_buffers.append(
                    _KDABuffers(
                        torch.empty_like(layer_state.q_conv),
                        torch.empty_like(layer_state.k_conv),
                        torch.empty_like(layer_state.v_conv),
                        torch.empty_like(layer_state.recurrent),
                    )
                )
            elif isinstance(layer_state, MLALayerState):
                if (
                    not layer_state.is_static
                    or layer_state.position is None
                    or layer_state.key_pass is None
                    or layer_state.value is None
                ):
                    raise ValueError("all MLA layers must use complete static state")
                if state.tokens_seen + max_speculative_tokens > layer_state.capacity:
                    raise ValueError("MLA cache has insufficient speculative capacity")
                offsets = torch.arange(
                    max_speculative_tokens,
                    dtype=torch.long,
                    device=layer_state.position.device,
                )
                indices = torch.empty_like(offsets)
                layer_buffers.append(
                    _MLABuffers(
                        offsets,
                        indices,
                        layer_state.compressed_kv.new_empty(
                            layer_state.compressed_kv.shape[0],
                            max_speculative_tokens,
                            layer_state.compressed_kv.shape[2],
                        ),
                        layer_state.rotary_key.new_empty(
                            layer_state.rotary_key.shape[0],
                            max_speculative_tokens,
                            layer_state.rotary_key.shape[2],
                        ),
                        layer_state.key_pass.new_empty(
                            layer_state.key_pass.shape[0],
                            layer_state.key_pass.shape[1],
                            max_speculative_tokens,
                            layer_state.key_pass.shape[3],
                        ),
                        layer_state.value.new_empty(
                            layer_state.value.shape[0],
                            layer_state.value.shape[1],
                            max_speculative_tokens,
                            layer_state.value.shape[3],
                        ),
                        layer_state.position.clone(),
                    )
                )
            else:
                raise TypeError("unsupported KLinear layer state")
        self._layer_buffers = tuple(layer_buffers)

        if state.attention_mask is None:
            self._mask_buffers = None
        else:
            if state.tokens_seen + max_speculative_tokens > state.attention_mask.shape[1]:
                raise ValueError("attention mask has insufficient speculative capacity")
            offsets = torch.arange(
                max_speculative_tokens,
                dtype=torch.long,
                device=state.position.device,
            )
            self._mask_buffers = _MaskBuffers(
                offsets,
                torch.empty_like(offsets),
                state.attention_mask.new_empty(
                    state.attention_mask.shape[0],
                    max_speculative_tokens,
                ),
            )

    @property
    def snapshot_bytes(self) -> int:
        """Return the bytes copied into checkpoint storage per snapshot."""

        tensors: list[torch.Tensor] = [self._position]
        for buffers in self._layer_buffers:
            if isinstance(buffers, _KDABuffers):
                tensors.extend(
                    (buffers.q_conv, buffers.k_conv, buffers.v_conv, buffers.recurrent)
                )
            elif isinstance(buffers, _MLABuffers):
                tensors.extend(
                    (
                        buffers.compressed_kv,
                        buffers.rotary_key,
                        buffers.key_pass,
                        buffers.value,
                        buffers.position,
                    )
                )
        if self._mask_buffers is not None:
            tensors.append(self._mask_buffers.values)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _validate_bound_state(self, state: KLinearDecodeState) -> None:
        if len(state.layer_states) != len(self._bound_layers) or any(
            current is not bound
            for current, bound in zip(
                state.layer_states,
                self._bound_layers,
                strict=True,
            )
        ):
            raise ValueError("checkpoint is bound to different layer state objects")
        if state.attention_mask is not self._bound_attention_mask:
            raise ValueError("checkpoint is bound to a different attention mask")
        if state.position is not self._bound_position:
            raise ValueError("checkpoint is bound to a different top-level position")
        for layer_state in state.layer_states:
            if isinstance(layer_state, MLALayerState):
                if state.tokens_seen + self.max_speculative_tokens > layer_state.capacity:
                    raise ValueError("MLA cache has insufficient speculative capacity")
        if (
            state.attention_mask is not None
            and state.tokens_seen + self.max_speculative_tokens
            > state.attention_mask.shape[1]
        ):
            raise ValueError("attention mask has insufficient speculative capacity")

    def snapshot(self, state: KLinearDecodeState) -> None:
        """Copy the current state into the already allocated buffers."""

        self._validate_bound_state(state)
        self._tokens_seen = state.tokens_seen
        self._position.copy_(state.position)
        for layer_state, buffers in zip(
            state.layer_states,
            self._layer_buffers,
            strict=True,
        ):
            if layer_state is None:
                continue
            if isinstance(layer_state, KDALayerState) and isinstance(
                buffers, _KDABuffers
            ):
                buffers.q_conv.copy_(layer_state.q_conv)
                buffers.k_conv.copy_(layer_state.k_conv)
                buffers.v_conv.copy_(layer_state.v_conv)
                buffers.recurrent.copy_(layer_state.recurrent)
            elif isinstance(layer_state, MLALayerState) and isinstance(
                buffers, _MLABuffers
            ):
                torch.add(layer_state.position, buffers.offsets, out=buffers.indices)
                torch.index_select(
                    layer_state.compressed_kv,
                    1,
                    buffers.indices,
                    out=buffers.compressed_kv,
                )
                torch.index_select(
                    layer_state.rotary_key,
                    1,
                    buffers.indices,
                    out=buffers.rotary_key,
                )
                torch.index_select(
                    layer_state.key_pass,
                    2,
                    buffers.indices,
                    out=buffers.key_pass,
                )
                torch.index_select(
                    layer_state.value,
                    2,
                    buffers.indices,
                    out=buffers.value,
                )
                buffers.position.copy_(layer_state.position)
            else:
                raise ValueError("checkpoint layer kinds changed")
        if self._mask_buffers is not None:
            torch.add(
                state.position,
                self._mask_buffers.offsets,
                out=self._mask_buffers.indices,
            )
            torch.index_select(
                state.attention_mask,
                1,
                self._mask_buffers.indices,
                out=self._mask_buffers.values,
            )

    def restore(self, state: KLinearDecodeState) -> KLinearDecodeState:
        """Restore the latest snapshot without changing captured tensor addresses."""

        self._validate_bound_state(state)
        for layer_state, buffers in zip(
            state.layer_states,
            self._layer_buffers,
            strict=True,
        ):
            if layer_state is None:
                continue
            if isinstance(layer_state, KDALayerState) and isinstance(
                buffers, _KDABuffers
            ):
                layer_state.q_conv.copy_(buffers.q_conv)
                layer_state.k_conv.copy_(buffers.k_conv)
                layer_state.v_conv.copy_(buffers.v_conv)
                layer_state.recurrent.copy_(buffers.recurrent)
            elif isinstance(layer_state, MLALayerState) and isinstance(
                buffers, _MLABuffers
            ):
                layer_state.compressed_kv.index_copy_(
                    1, buffers.indices, buffers.compressed_kv
                )
                layer_state.rotary_key.index_copy_(1, buffers.indices, buffers.rotary_key)
                layer_state.key_pass.index_copy_(2, buffers.indices, buffers.key_pass)
                layer_state.value.index_copy_(2, buffers.indices, buffers.value)
                layer_state.position.copy_(buffers.position)
            else:
                raise ValueError("checkpoint layer kinds changed")
        if self._mask_buffers is not None:
            state.attention_mask.index_copy_(
                1,
                self._mask_buffers.indices,
                self._mask_buffers.values,
            )
        state.position.copy_(self._position)
        return state.with_tokens_seen(self._tokens_seen)
