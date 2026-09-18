"""Pinned-host storage and in-place restore for Kimi-Linear decode state.

MLA snapshots deliberately store less than the live state. They retain only
``compressed_kv``, ``rotary_key``, and device position, then rebuild projected
``key_pass`` and ``value`` buffers during restore through the model's existing
layernorm and ``kv_b_proj`` path. The real model stores 8,064 bytes per token in
this compressed form instead of 122,752 live bytes per token. Paying roughly
ten milliseconds of reasoned 32K rebuild work is intentional because storing
the projections would use about fifteen times the host memory merely to avoid
that rebuild against a prefill that takes seconds.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from engine.klinear.attention import MLAAttention
from engine.klinear.state import KDALayerState, KLinearDecodeState, MLALayerState

from .key import ModelFingerprint, fingerprint_digest, token_tuple


@dataclass(frozen=True)
class _HostKDA:
    q_conv: torch.Tensor
    k_conv: torch.Tensor
    v_conv: torch.Tensor
    recurrent: torch.Tensor


@dataclass(frozen=True)
class _HostMLA:
    compressed_kv: torch.Tensor
    rotary_key: torch.Tensor
    position: torch.Tensor


_HostLayer = _HostKDA | _HostMLA | None


@dataclass
class _Snapshot:
    layers: tuple[_HostLayer, ...]
    token_count: int
    attention_mask: torch.Tensor | None
    position: torch.Tensor
    token_ids: tuple[int, ...] | None
    model_digest: str | None
    byte_count: int
    ready_event: torch.cuda.Event | None = None


@dataclass(frozen=True)
class _DiskEntry:
    path: Path
    token_count: int
    token_ids: tuple[int, ...] | None
    model_digest: str | None
    byte_count: int


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _host_tensor(source: torch.Tensor) -> torch.Tensor:
    pin_memory = torch.cuda.is_available()
    target = torch.empty(
        source.shape,
        dtype=source.dtype,
        device="cpu",
        pin_memory=pin_memory,
    )
    target.copy_(source, non_blocking=source.device.type == "cuda" and pin_memory)
    return target


def _pinned_copy(source: torch.Tensor) -> torch.Tensor:
    target = torch.empty(
        source.shape,
        dtype=source.dtype,
        device="cpu",
        pin_memory=torch.cuda.is_available(),
    )
    target.copy_(source)
    return target


def _all_tensors(snapshot: _Snapshot) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = [snapshot.position]
    for layer in snapshot.layers:
        if isinstance(layer, _HostKDA):
            tensors.extend((layer.q_conv, layer.k_conv, layer.v_conv, layer.recurrent))
        elif isinstance(layer, _HostMLA):
            tensors.extend(
                (
                    layer.compressed_kv,
                    layer.rotary_key,
                    layer.position,
                )
            )
    if snapshot.attention_mask is not None:
        tensors.append(snapshot.attention_mask)
    return tuple(tensors)


class StateCache:
    """LRU storage for compressed prefix snapshots in pinned host memory.

    A live MLA state contains compressed latent and rotary tensors plus the much
    larger projected ``key_pass`` and ``value`` tensors. Snapshots deliberately
    store only the compressed latent, rotary key, and device position. Restore
    then repeats the existing MLA layernorm, ``kv_b_proj``, transpose, and split
    path to rebuild projected keys and values into the graph-bound buffers.

    For the real seven MLA layers this stores 8,064 bytes per token instead of
    122,752 bytes per token, about fifteen times less host memory. The trade is
    one projection per MLA layer over the cached prefix, reasoned at roughly
    0.96 TFLOP or about ten milliseconds at 32K. This is intentional because
    storing projected keys and values would spend fifteen times the bytes to
    save about ten milliseconds against a prefill that takes seconds.

    Pinned memory is required for genuinely asynchronous host-to-device copies.
    Pageable memory forces CUDA to stage through an internal pinned allocation,
    which adds an extra copy and can serialize the request hot path. CPU-only
    builds use ordinary host tensors because no CUDA pin allocator exists.

    ``load`` always copies into a caller-owned, fixed-capacity device state. It
    never replaces the target tensors, so their addresses remain the addresses
    used by a captured CUDA graph. ``allocate_state`` is a convenience for
    uncaptured callers and for capture setup. A serving integration with an
    existing graph should retain its graph-bound state and pass that to
    ``load`` on every request.

    Disk spill is off by default. When enabled explicitly, evicted snapshots
    are serialized to ``spill_directory``. A disk hit performs synchronous I/O
    and a synchronous restore because it belongs to a different latency class.
    """

    def __init__(
        self,
        byte_budget: int,
        *,
        disk_spill: bool = False,
        spill_directory: str | Path | None = None,
    ) -> None:
        if isinstance(byte_budget, bool) or not isinstance(byte_budget, int):
            raise TypeError("byte_budget must be an integer")
        if byte_budget < 0:
            raise ValueError("byte_budget cannot be negative")
        if disk_spill and spill_directory is None:
            raise ValueError("disk spill requires an explicit spill_directory")
        if not disk_spill and spill_directory is not None:
            raise ValueError("spill_directory requires disk_spill=True")

        self.byte_budget = byte_budget
        self.disk_spill = disk_spill
        self.spill_directory = (
            Path(spill_directory).resolve() if spill_directory is not None else None
        )
        if self.spill_directory is not None:
            self.spill_directory.mkdir(parents=True, exist_ok=True)
        self._entries: OrderedDict[str, _Snapshot] = OrderedDict()
        self._disk_entries: OrderedDict[str, _DiskEntry] = OrderedDict()
        self._bytes = 0

    @property
    def host_bytes(self) -> int:
        return self._bytes

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def disk_entry_count(self) -> int:
        return len(self._disk_entries)

    def __len__(self) -> int:
        return len(self._entries) + len(self._disk_entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries or key in self._disk_entries

    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries) + tuple(
            key for key in self._disk_entries if key not in self._entries
        )

    def entry_bytes(self, key: str) -> int | None:
        snapshot = self._entries.get(key)
        if snapshot is not None:
            return snapshot.byte_count
        disk = self._disk_entries.get(key)
        return None if disk is None else disk.byte_count

    def _estimate_bytes(self, state: KLinearDecodeState, token_count: int) -> int:
        total = torch.empty((), dtype=torch.long).element_size()
        for layer in state.layer_states:
            if layer is None:
                continue
            if isinstance(layer, KDALayerState):
                total += sum(
                    _tensor_bytes(tensor)
                    for tensor in (
                        layer.q_conv,
                        layer.k_conv,
                        layer.v_conv,
                        layer.recurrent,
                    )
                )
            elif isinstance(layer, MLALayerState):
                if token_count > layer.capacity:
                    raise ValueError("MLA state is shorter than token_count")
                total += _tensor_bytes(layer.compressed_kv[:, :token_count])
                total += _tensor_bytes(layer.rotary_key[:, :token_count])
                total += torch.empty((), dtype=torch.long).element_size()
            else:
                raise TypeError("unsupported KLinear layer state")
        if state.attention_mask is not None:
            if token_count > state.attention_mask.shape[1]:
                raise ValueError("attention mask is shorter than token_count")
            total += _tensor_bytes(state.attention_mask[:, :token_count])
        return total

    def _make_snapshot(
        self,
        state: KLinearDecodeState,
        token_count: int,
        *,
        token_ids: tuple[int, ...] | None,
        model_digest: str | None,
    ) -> _Snapshot:
        layers: list[_HostLayer] = []
        cuda_device: torch.device | None = None
        for layer in state.layer_states:
            if layer is None:
                layers.append(None)
            elif isinstance(layer, KDALayerState):
                if layer.recurrent.device.type == "cuda":
                    cuda_device = layer.recurrent.device
                layers.append(
                    _HostKDA(
                        _host_tensor(layer.q_conv),
                        _host_tensor(layer.k_conv),
                        _host_tensor(layer.v_conv),
                        _host_tensor(layer.recurrent),
                    )
                )
            elif isinstance(layer, MLALayerState):
                if token_count > layer.capacity:
                    raise ValueError("MLA state is shorter than token_count")
                if layer.compressed_kv.device.type == "cuda":
                    cuda_device = layer.compressed_kv.device
                layer_position = (
                    layer.position
                    if layer.position is not None
                    else torch.full(
                        (),
                        token_count,
                        dtype=torch.long,
                        device=layer.compressed_kv.device,
                    )
                )
                layers.append(
                    _HostMLA(
                        _host_tensor(layer.compressed_kv[:, :token_count]),
                        _host_tensor(layer.rotary_key[:, :token_count]),
                        _host_tensor(layer_position),
                    )
                )
            else:
                raise TypeError("unsupported KLinear layer state")

        attention_mask = (
            None
            if state.attention_mask is None
            else _host_tensor(state.attention_mask[:, :token_count])
        )
        position_source = (
            state.position
            if state.position is not None
            else torch.full(
                (),
                token_count,
                dtype=torch.long,
                device=cuda_device or torch.device("cpu"),
            )
        )
        position = _host_tensor(position_source)
        snapshot = _Snapshot(
            tuple(layers),
            token_count,
            attention_mask,
            position,
            token_ids,
            model_digest,
            0,
        )
        snapshot.byte_count = sum(_tensor_bytes(tensor) for tensor in _all_tensors(snapshot))
        if cuda_device is not None:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(cuda_device))
            snapshot.ready_event = event
        return snapshot

    def _wait_until_ready(self, snapshot: _Snapshot, device: torch.device | None) -> None:
        event = snapshot.ready_event
        if event is None:
            return
        if device is not None and device.type == "cuda":
            torch.cuda.current_stream(device).wait_event(event)
        else:
            event.synchronize()

    def _delete_existing(self, key: str) -> None:
        snapshot = self._entries.pop(key, None)
        if snapshot is not None:
            self._wait_until_ready(snapshot, None)
            self._bytes -= snapshot.byte_count
        disk = self._disk_entries.pop(key, None)
        if disk is not None and disk.path.exists():
            disk.path.unlink()

    def save(
        self,
        key: str,
        state: KLinearDecodeState,
        token_count: int,
        *,
        token_ids: torch.Tensor | tuple[int, ...] | list[int] | None = None,
        model_fingerprint: ModelFingerprint | None = None,
    ) -> bool:
        """Snapshot the live prefix and enforce the host-memory byte budget."""

        if not key:
            raise ValueError("cache key cannot be empty")
        if token_count != state.tokens_seen:
            raise ValueError("token_count must equal state.tokens_seen")
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        normalized_tokens = None if token_ids is None else token_tuple(token_ids)
        if normalized_tokens is not None and len(normalized_tokens) != token_count:
            raise ValueError("token_ids length must equal token_count")
        model_digest = (
            None
            if model_fingerprint is None
            else fingerprint_digest(model_fingerprint)
        )
        estimated = self._estimate_bytes(state, token_count)
        self._delete_existing(key)
        if estimated > self.byte_budget and not self.disk_spill:
            return False

        snapshot = self._make_snapshot(
            state,
            token_count,
            token_ids=normalized_tokens,
            model_digest=model_digest,
        )
        self._entries[key] = snapshot
        self._entries.move_to_end(key)
        self._bytes += snapshot.byte_count
        while self._bytes > self.byte_budget and self._entries:
            self.evict()
        return key in self

    def _disk_path(self, key: str) -> Path:
        if self.spill_directory is None:
            raise RuntimeError("disk spill is disabled")
        safe_name = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.spill_directory / f"{safe_name}.pt"

    def _snapshot_payload(self, snapshot: _Snapshot) -> dict[str, Any]:
        layers: list[dict[str, Any] | None] = []
        for layer in snapshot.layers:
            if layer is None:
                layers.append(None)
            elif isinstance(layer, _HostKDA):
                layers.append(
                    {
                        "kind": "kda",
                        "q_conv": layer.q_conv,
                        "k_conv": layer.k_conv,
                        "v_conv": layer.v_conv,
                        "recurrent": layer.recurrent,
                    }
                )
            else:
                layers.append(
                    {
                        "kind": "mla",
                        "compressed_kv": layer.compressed_kv,
                        "rotary_key": layer.rotary_key,
                        "position": layer.position,
                    }
                )
        return {
            "schema": 2,
            "layers": layers,
            "token_count": snapshot.token_count,
            "attention_mask": snapshot.attention_mask,
            "position": snapshot.position,
            "token_ids": snapshot.token_ids,
            "model_digest": snapshot.model_digest,
            "byte_count": snapshot.byte_count,
        }

    def _snapshot_from_payload(self, payload: dict[str, Any]) -> _Snapshot:
        if payload.get("schema") != 2:
            raise ValueError("unsupported state-cache disk snapshot schema")
        layers: list[_HostLayer] = []
        for raw in payload["layers"]:
            if raw is None:
                layers.append(None)
            elif raw["kind"] == "kda":
                layers.append(
                    _HostKDA(
                        _pinned_copy(raw["q_conv"]),
                        _pinned_copy(raw["k_conv"]),
                        _pinned_copy(raw["v_conv"]),
                        _pinned_copy(raw["recurrent"]),
                    )
                )
            elif raw["kind"] == "mla":
                layers.append(
                    _HostMLA(
                        _pinned_copy(raw["compressed_kv"]),
                        _pinned_copy(raw["rotary_key"]),
                        _pinned_copy(raw["position"]),
                    )
                )
            else:
                raise ValueError("invalid state-cache disk layer kind")
        mask = payload["attention_mask"]
        return _Snapshot(
            tuple(layers),
            int(payload["token_count"]),
            None if mask is None else _pinned_copy(mask),
            _pinned_copy(payload["position"]),
            None
            if payload["token_ids"] is None
            else tuple(int(value) for value in payload["token_ids"]),
            payload["model_digest"],
            int(payload["byte_count"]),
        )

    def _write_to_disk(self, key: str, snapshot: _Snapshot) -> None:
        path = self._disk_path(key)
        torch.save(self._snapshot_payload(snapshot), path)
        self._disk_entries[key] = _DiskEntry(
            path,
            snapshot.token_count,
            snapshot.token_ids,
            snapshot.model_digest,
            snapshot.byte_count,
        )
        self._disk_entries.move_to_end(key)

    def _read_from_disk(self, key: str) -> _Snapshot:
        entry = self._disk_entries[key]
        payload = torch.load(
            entry.path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise ValueError("invalid state-cache disk snapshot")
        self._disk_entries.move_to_end(key)
        return self._snapshot_from_payload(payload)

    def evict(self) -> str | None:
        """Evict the least recently used host snapshot, optionally to disk."""

        if not self._entries:
            return None
        key, snapshot = self._entries.popitem(last=False)
        self._wait_until_ready(snapshot, None)
        self._bytes -= snapshot.byte_count
        if self.disk_spill:
            self._write_to_disk(key, snapshot)
        return key

    def _snapshot_for_key(self, key: str) -> tuple[_Snapshot, bool] | None:
        snapshot = self._entries.get(key)
        if snapshot is not None:
            self._entries.move_to_end(key)
            return snapshot, True
        if key in self._disk_entries:
            return self._read_from_disk(key), False
        return None

    @staticmethod
    def _state_device(state: KLinearDecodeState) -> torch.device:
        if state.position is not None:
            return state.position.device
        for layer in state.layer_states:
            if isinstance(layer, KDALayerState):
                return layer.recurrent.device
            if isinstance(layer, MLALayerState):
                return layer.compressed_kv.device
        if state.attention_mask is not None:
            return state.attention_mask.device
        return torch.device("cpu")

    @staticmethod
    def _validate_target(snapshot: _Snapshot, target: KLinearDecodeState) -> None:
        if not target.is_static or target.position is None:
            raise ValueError("load requires a fixed-capacity target state")
        if len(snapshot.layers) != len(target.layer_states):
            raise ValueError("snapshot and target layer counts disagree")
        for saved, current in zip(snapshot.layers, target.layer_states, strict=True):
            if saved is None or current is None:
                if saved is not None or current is not None:
                    raise ValueError("snapshot and target layer kinds disagree")
            elif isinstance(saved, _HostKDA) and isinstance(current, KDALayerState):
                if not current.is_static:
                    raise ValueError("target KDA layer is not fixed-capacity")
                pairs = (
                    (saved.q_conv, current.q_conv),
                    (saved.k_conv, current.k_conv),
                    (saved.v_conv, current.v_conv),
                    (saved.recurrent, current.recurrent),
                )
                if any(
                    source.shape != value.shape or source.dtype != value.dtype
                    for source, value in pairs
                ):
                    raise ValueError("snapshot and target KDA shapes or dtypes disagree")
            elif isinstance(saved, _HostMLA) and isinstance(current, MLALayerState):
                if (
                    not current.is_static
                    or current.position is None
                    or current.key_pass is None
                    or current.value is None
                ):
                    raise ValueError("target MLA layer is incomplete")
                if current.capacity < snapshot.token_count:
                    raise ValueError("target MLA capacity is shorter than the snapshot")
                pairs = (
                    (saved.compressed_kv, current.compressed_kv[:, : snapshot.token_count]),
                    (saved.rotary_key, current.rotary_key[:, : snapshot.token_count]),
                )
                if any(
                    source.shape != value.shape or source.dtype != value.dtype
                    for source, value in pairs
                ):
                    raise ValueError("snapshot and target MLA shapes or dtypes disagree")
            else:
                raise ValueError("snapshot and target layer kinds disagree")
        if (snapshot.attention_mask is None) != (target.attention_mask is None):
            raise ValueError("snapshot and target attention-mask kinds disagree")
        if target.attention_mask is not None:
            if target.attention_mask.shape[1] < snapshot.token_count:
                raise ValueError("target attention-mask capacity is too small")
            target_prefix = target.attention_mask[:, : snapshot.token_count]
            if (
                snapshot.attention_mask.shape != target_prefix.shape
                or snapshot.attention_mask.dtype != target_prefix.dtype
            ):
                raise ValueError("snapshot and target attention masks disagree")

    def _copy_snapshot(
        self,
        snapshot: _Snapshot,
        target: KLinearDecodeState,
        *,
        non_blocking: bool,
    ) -> None:
        self._validate_target(snapshot, target)
        for saved, current in zip(snapshot.layers, target.layer_states, strict=True):
            if saved is None:
                continue
            if isinstance(saved, _HostKDA) and isinstance(current, KDALayerState):
                current.q_conv.copy_(saved.q_conv, non_blocking=non_blocking)
                current.k_conv.copy_(saved.k_conv, non_blocking=non_blocking)
                current.v_conv.copy_(saved.v_conv, non_blocking=non_blocking)
                current.recurrent.copy_(saved.recurrent, non_blocking=non_blocking)
            elif isinstance(saved, _HostMLA) and isinstance(current, MLALayerState):
                count = snapshot.token_count
                current.compressed_kv[:, :count].copy_(
                    saved.compressed_kv, non_blocking=non_blocking
                )
                current.rotary_key[:, :count].copy_(
                    saved.rotary_key, non_blocking=non_blocking
                )
                # Static MLA scores the full capacity before adding its finite
                # mask. NaNs in an uninitialized tail therefore survive masking.
                current.compressed_kv[:, count:].zero_()
                current.rotary_key[:, count:].zero_()
                current.key_pass[:, :, count:].zero_()
                current.value[:, :, count:].zero_()
                current.position.copy_(saved.position, non_blocking=non_blocking)
        if snapshot.attention_mask is not None:
            target.attention_mask[:, : snapshot.token_count].copy_(
                snapshot.attention_mask,
                non_blocking=non_blocking,
            )
            target.attention_mask[:, snapshot.token_count :].zero_()
        target.position.copy_(snapshot.position, non_blocking=non_blocking)
        object.__setattr__(target, "tokens_seen", snapshot.token_count)

    @staticmethod
    def _model_layers(model: Any, expected_count: int) -> Any:
        layers = getattr(model, "layers", None)
        if layers is None or len(layers) != expected_count:
            raise ValueError("model layers do not match the cached state")
        return layers

    @torch.inference_mode()
    def rebuild_mla(
        self,
        model: Any,
        state: KLinearDecodeState,
        *,
        token_count: int | None = None,
    ) -> None:
        """Rebuild projected MLA keys and values into existing device buffers.

        This is the same path used by ``MLAAttention`` when projected caches are
        absent: layernorm the compressed latent, apply ``kv_b_proj``, reshape,
        transpose, then split key and value widths. Only the live prefix is
        rebuilt because fixed-capacity tails are not part of the snapshot.
        """

        count = state.tokens_seen if token_count is None else token_count
        if count < 0 or count > state.tokens_seen:
            raise ValueError("rebuild token_count is outside the live state")
        model_layers = self._model_layers(model, len(state.layer_states))
        for model_layer, layer_state in zip(
            model_layers,
            state.layer_states,
            strict=True,
        ):
            if not isinstance(layer_state, MLALayerState):
                continue
            attention = getattr(model_layer, "self_attn", None)
            if not isinstance(attention, MLAAttention):
                raise ValueError("model and state disagree on an MLA layer")
            if layer_state.key_pass is None or layer_state.value is None:
                raise ValueError("target MLA layer has no projected buffers")
            if layer_state.capacity < count:
                raise ValueError("target MLA capacity is shorter than the rebuild")

            latent = layer_state.compressed_kv[:, :count]
            batch = latent.shape[0]
            rebuilt = attention.kv_b_proj(attention.kv_a_layernorm(latent)).view(
                batch,
                count,
                attention.num_heads,
                attention.qk_nope_head_dim + attention.v_head_dim,
            )
            rebuilt = rebuilt.transpose(1, 2)
            cached_key_pass, cached_value = torch.split(
                rebuilt,
                [attention.qk_nope_head_dim, attention.v_head_dim],
                dim=-1,
            )
            layer_state.key_pass[:, :, :count].copy_(cached_key_pass)
            layer_state.value[:, :, :count].copy_(cached_value)

    def load(
        self,
        key: str,
        into_state: KLinearDecodeState,
        *,
        model: Any,
    ) -> bool:
        """Restore compressed state, rebuild MLA projections, and report a hit."""

        found = self._snapshot_for_key(key)
        if found is None:
            return False
        snapshot, in_memory = found
        device = self._state_device(into_state)
        self._wait_until_ready(snapshot, device)
        non_blocking = in_memory and device.type == "cuda"
        self._copy_snapshot(snapshot, into_state, non_blocking=non_blocking)
        self.rebuild_mla(model, into_state, token_count=snapshot.token_count)
        if not in_memory and device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        return True

    def allocate_state(
        self,
        key: str,
        *,
        model: Any,
        device: torch.device | str,
        additional_tokens: int = 1,
    ) -> KLinearDecodeState:
        """Allocate a compatible restore target before calling ``load``."""

        if additional_tokens < 0:
            raise ValueError("additional_tokens cannot be negative")
        found = self._snapshot_for_key(key)
        if found is None:
            raise KeyError(key)
        snapshot, _ = found
        self._wait_until_ready(snapshot, None)
        device = torch.device(device)
        capacity = snapshot.token_count + additional_tokens
        model_layers = self._model_layers(model, len(snapshot.layers))
        layers: list[KDALayerState | MLALayerState | None] = []
        for model_layer, saved in zip(model_layers, snapshot.layers, strict=True):
            if saved is None:
                layers.append(None)
            elif isinstance(saved, _HostKDA):
                layers.append(
                    KDALayerState(
                        torch.empty(saved.q_conv.shape, dtype=saved.q_conv.dtype, device=device),
                        torch.empty(saved.k_conv.shape, dtype=saved.k_conv.dtype, device=device),
                        torch.empty(saved.v_conv.shape, dtype=saved.v_conv.dtype, device=device),
                        torch.empty(
                            saved.recurrent.shape,
                            dtype=saved.recurrent.dtype,
                            device=device,
                        ),
                        is_static=True,
                    )
                )
            else:
                batch, _, latent_width = saved.compressed_kv.shape
                rotary_width = saved.rotary_key.shape[-1]
                attention = getattr(model_layer, "self_attn", None)
                if not isinstance(attention, MLAAttention):
                    raise ValueError("model and snapshot disagree on an MLA layer")
                heads = attention.num_heads
                key_width = attention.qk_nope_head_dim
                value_width = attention.v_head_dim
                layers.append(
                    MLALayerState(
                        torch.empty(
                            batch,
                            capacity,
                            latent_width,
                            dtype=saved.compressed_kv.dtype,
                            device=device,
                        ),
                        torch.empty(
                            batch,
                            capacity,
                            rotary_width,
                            dtype=saved.rotary_key.dtype,
                            device=device,
                        ),
                        torch.empty(
                            batch,
                            heads,
                            capacity,
                            key_width,
                            dtype=saved.compressed_kv.dtype,
                            device=device,
                        ),
                        torch.empty(
                            batch,
                            heads,
                            capacity,
                            value_width,
                            dtype=saved.compressed_kv.dtype,
                            device=device,
                        ),
                        torch.full(
                            (),
                            snapshot.token_count,
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                )
        attention_mask = None
        if snapshot.attention_mask is not None:
            attention_mask = torch.empty(
                snapshot.attention_mask.shape[0],
                capacity,
                dtype=snapshot.attention_mask.dtype,
                device=device,
            )
        return KLinearDecodeState(
            tuple(layers),
            snapshot.token_count,
            attention_mask,
            torch.full(
                (),
                snapshot.token_count,
                dtype=torch.long,
                device=device,
            ),
        )

    def find_longest_prefix(
        self,
        token_ids: torch.Tensor | tuple[int, ...] | list[int],
        model_fingerprint: ModelFingerprint,
    ) -> tuple[str, int] | None:
        """Find the longest indexed exact token prefix for one model."""

        tokens = token_tuple(token_ids)
        wanted_digest = fingerprint_digest(model_fingerprint)
        candidates: list[tuple[str, int, tuple[int, ...] | None, str | None]] = []
        candidates.extend(
            (key, item.token_count, item.token_ids, item.model_digest)
            for key, item in self._entries.items()
        )
        candidates.extend(
            (key, item.token_count, item.token_ids, item.model_digest)
            for key, item in self._disk_entries.items()
            if key not in self._entries
        )
        candidates.sort(key=lambda item: item[1], reverse=True)
        for key, count, saved_tokens, saved_digest in candidates:
            if count > len(tokens) or saved_digest != wanted_digest:
                continue
            if saved_tokens is not None and tokens[:count] == saved_tokens:
                return key, count
        return None

    def synchronize(self, key: str) -> None:
        """Wait for an asynchronous device-to-host snapshot to finish."""

        snapshot = self._entries.get(key)
        if snapshot is not None:
            self._wait_until_ready(snapshot, None)

    def clear(self) -> None:
        """Remove every host and disk snapshot owned by this cache."""

        for snapshot in self._entries.values():
            self._wait_until_ready(snapshot, None)
        self._entries.clear()
        self._bytes = 0
        for entry in self._disk_entries.values():
            if entry.path.exists():
                entry.path.unlink()
        self._disk_entries.clear()
