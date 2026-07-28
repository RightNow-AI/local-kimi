"""Plain PyTorch KDA and MLA attention paths for Kimi K3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSGatedNorm, RMSNorm


@dataclass
class KDAState:
    q_conv: torch.Tensor
    k_conv: torch.Tensor
    v_conv: torch.Tensor
    # Moonshot requests the V-first layout through transpose_state_layout=True.
    recurrent: torch.Tensor


@dataclass
class MLAState:
    key: torch.Tensor
    value: torch.Tensor


class DepthwiseShortConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(hidden_size, 1, kernel_size, device=device, dtype=dtype)
        )
        nn.init.normal_(self.weight, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, channels = values.shape
        if state is None:
            state = values.new_zeros(batch, channels, self.kernel_size)
        if state.shape != (batch, channels, self.kernel_size):
            raise ValueError("short convolution state has an invalid shape")

        outputs: list[torch.Tensor] = []
        cache = state
        weights = self.weight[:, 0, :]
        for token_index in range(sequence):
            token = values[:, token_index]
            shifted = torch.cat((cache[:, :, 1:], token.unsqueeze(-1)), dim=-1)
            if attention_mask is not None:
                active = attention_mask[:, token_index].bool().view(batch, 1, 1)
                cache = torch.where(active, shifted, cache)
            else:
                active = None
                cache = shifted
            convolved = (cache.float() * weights.float().unsqueeze(0)).sum(dim=-1)
            convolved = F.silu(convolved).to(values.dtype)
            if active is not None:
                convolved = torch.where(active.squeeze(-1), convolved, torch.zeros_like(convolved))
            outputs.append(convolved)
        return torch.stack(outputs, dim=1), cache


class KDAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        conv_size: int = 4,
        gate_lower_bound: float | None = -5.0,
        rms_norm_eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        projection_size = num_heads * head_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = projection_size
        self.gate_lower_bound = gate_lower_bound

        self.q_proj = nn.Linear(hidden_size, projection_size, bias=False, **factory)
        self.k_proj = nn.Linear(hidden_size, projection_size, bias=False, **factory)
        self.v_proj = nn.Linear(hidden_size, projection_size, bias=False, **factory)
        self.q_conv1d = DepthwiseShortConv(
            projection_size, conv_size, device=device, dtype=torch.float32
        )
        self.k_conv1d = DepthwiseShortConv(
            projection_size, conv_size, device=device, dtype=torch.float32
        )
        self.v_conv1d = DepthwiseShortConv(
            projection_size, conv_size, device=device, dtype=torch.float32
        )
        # Real K3 weights carry one decay rate per head dimension, shared by heads.
        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(head_dim, device=device, dtype=torch.float32).uniform_(1, 16)
            )
        )
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **factory)
        self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False, **factory)
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, device=device, dtype=torch.float32)
        )
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False, **factory)
        self.g_proj = nn.Linear(hidden_size, projection_size, bias=False, **factory)
        self.o_norm = RMSGatedNorm(
            head_dim, eps=rms_norm_eps, device=device, dtype=torch.float32
        )
        self.o_proj = nn.Linear(projection_size, hidden_size, bias=False, **factory)

    def _decay_gate(self, raw_gate: torch.Tensor) -> torch.Tensor:
        biased = raw_gate.float() + self.dt_bias.float().view(
            self.num_heads, self.head_dim
        )
        rate = self.A_log.float().exp().view(1, self.head_dim)
        if self.gate_lower_bound is not None:
            return self.gate_lower_bound * torch.sigmoid(rate * biased)
        return -rate * F.softplus(biased)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: KDAState | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KDAState]:
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("KDA attention_mask must have shape [batch, sequence]")
        batch, sequence, _ = hidden_states.shape
        q_state = state.q_conv if state is not None else None
        k_state = state.k_conv if state is not None else None
        v_state = state.v_conv if state is not None else None

        q, q_state = self.q_conv1d(
            self.q_proj(hidden_states), q_state, attention_mask
        )
        k, k_state = self.k_conv1d(
            self.k_proj(hidden_states), k_state, attention_mask
        )
        v, v_state = self.v_conv1d(
            self.v_proj(hidden_states), v_state, attention_mask
        )

        raw_decay = self.f_b_proj(self.f_a_proj(hidden_states)).view(
            batch, sequence, self.num_heads, self.head_dim
        )
        beta = torch.sigmoid(self.b_proj(hidden_states).float())
        q = q.view(batch, sequence, self.num_heads, self.head_dim).float()
        k = k.view(batch, sequence, self.num_heads, self.head_dim).float()
        v = v.view(batch, sequence, self.num_heads, self.head_dim).float()
        q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
        decay = self._decay_gate(raw_decay)

        if state is None:
            recurrent = torch.zeros(
                batch,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                dtype=torch.float32,
                device=hidden_states.device,
            )
        else:
            recurrent = state.recurrent.float()
        if recurrent.shape != (
            batch,
            self.num_heads,
            self.head_dim,
            self.head_dim,
        ):
            raise ValueError("KDA recurrent state has an invalid shape")

        outputs: list[torch.Tensor] = []
        scale = self.head_dim**-0.5
        for token_index in range(sequence):
            q_token = q[:, token_index]
            k_token = k[:, token_index]
            v_token = v[:, token_index]
            decay_token = decay[:, token_index]
            beta_token = beta[:, token_index]

            candidate = recurrent * torch.exp(decay_token).unsqueeze(-2)
            prediction = (candidate * k_token.unsqueeze(-2)).sum(dim=-1)
            delta = v_token - prediction
            candidate = candidate + (
                beta_token.unsqueeze(-1).unsqueeze(-1)
                * delta.unsqueeze(-1)
                * k_token.unsqueeze(-2)
            )
            output = (candidate * (q_token * scale).unsqueeze(-2)).sum(dim=-1)
            if attention_mask is not None:
                active = attention_mask[:, token_index].bool().view(batch, 1, 1)
                recurrent = torch.where(active.unsqueeze(-1), candidate, recurrent)
                output = torch.where(active, output, torch.zeros_like(output))
            else:
                recurrent = candidate
            outputs.append(output)

        output = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        output_gate = self.g_proj(hidden_states).view(
            batch, sequence, self.num_heads, self.head_dim
        )
        output = self.o_norm(output, output_gate)
        output = self.o_proj(output.reshape(batch, sequence, self.projection_size))
        final_state = KDAState(q_state, k_state, v_state, recurrent)
        if return_state:
            return output, final_state
        return output


class MLAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        *,
        use_output_gate: bool = True,
        rms_norm_eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.scaling = self.q_head_dim**-0.5
        self.use_output_gate = use_output_gate

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False, **factory)
        self.q_a_layernorm = RMSNorm(
            q_lora_rank, eps=rms_norm_eps, device=device, dtype=dtype
        )
        self.q_b_proj = nn.Linear(
            q_lora_rank, num_heads * self.q_head_dim, bias=False, **factory
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False, **factory
        )
        self.kv_a_layernorm = RMSNorm(
            kv_lora_rank, eps=rms_norm_eps, device=device, dtype=dtype
        )
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            **factory,
        )
        self.o_proj = nn.Linear(
            num_heads * v_head_dim, hidden_size, bias=False, **factory
        )
        self.g_proj = (
            nn.Linear(hidden_size, num_heads * v_head_dim, bias=False, **factory)
            if use_output_gate
            else None
        )

    def _attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch: int,
        query_length: int,
        key_length: int,
        past_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask.float()
        query_positions = past_length + torch.arange(query_length, device=device)
        key_positions = torch.arange(key_length, device=device)
        allowed = key_positions.view(1, 1, 1, key_length) <= query_positions.view(
            1, 1, query_length, 1
        )
        if attention_mask is not None and attention_mask.ndim == 2:
            if attention_mask.shape[-1] != key_length:
                raise ValueError("MLA padding mask must cover the complete key sequence")
            allowed = allowed & attention_mask[:, None, None, :].bool()
        additive = torch.zeros(
            (batch, 1, query_length, key_length),
            dtype=torch.float32,
            device=device,
        )
        additive.masked_fill_(~allowed, torch.finfo(torch.float32).min)
        return additive

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: MLAState | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MLAState]:
        batch, sequence, _ = hidden_states.shape
        query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query = query.view(batch, sequence, self.num_heads, self.q_head_dim).transpose(1, 2)
        query_pass, query_rot = torch.split(
            query, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        key_pass, key_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        key_pass = self.kv_b_proj(self.kv_a_layernorm(key_pass)).view(
            batch,
            sequence,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(1, 2)
        key_pass, value = torch.split(
            key_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        key_rot = key_rot.view(batch, 1, sequence, self.qk_rope_head_dim)
        key_rot = key_rot.expand(*key_pass.shape[:-1], -1)

        # Moonshot's provided implementation concatenates these raw rotary slices.
        query = torch.cat((query_pass, query_rot), dim=-1)
        key = torch.cat((key_pass, key_rot), dim=-1)
        past_length = 0
        if state is not None:
            past_length = state.key.shape[-2]
            key = torch.cat((state.key, key), dim=-2)
            value = torch.cat((state.value, value), dim=-2)

        mask = self._attention_mask(
            attention_mask,
            batch,
            sequence,
            key.shape[-2],
            past_length,
            hidden_states.device,
        )
        scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * self.scaling
        scores = scores.float() + mask
        probabilities = scores.softmax(dim=-1).to(query.dtype)
        output = torch.einsum("bhqk,bhkd->bhqd", probabilities, value)
        output = output.transpose(1, 2).reshape(
            batch, sequence, self.num_heads * self.v_head_dim
        )
        if self.g_proj is not None:
            output = output * self.g_proj(hidden_states).sigmoid()
        output = self.o_proj(output)
        final_state = MLAState(key, value)
        if return_state:
            return output, final_state
        return output
