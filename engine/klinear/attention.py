"""Pure PyTorch KDA and MLA attention for Kimi-Linear."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSGatedNorm, RMSNorm
from .state import KDALayerState, MLALayerState


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
        expected = (batch, channels, self.kernel_size)
        if tuple(state.shape) != expected:
            raise ValueError(
                f"short convolution state has shape {tuple(state.shape)}, expected {expected}"
            )

        outputs: list[torch.Tensor] = []
        cache = state
        weights = self.weight[:, 0, :]
        for token_index in range(sequence):
            token = values[:, token_index]
            shifted = torch.cat((cache[:, :, 1:], token.unsqueeze(-1)), dim=-1)
            active = None
            if attention_mask is not None:
                active = attention_mask[:, token_index].bool().view(batch, 1, 1)
                cache = torch.where(active, shifted, cache)
            else:
                cache = shifted
            convolved = (cache.float() * weights.float().unsqueeze(0)).sum(dim=-1)
            convolved = F.silu(convolved).to(values.dtype)
            if active is not None:
                convolved = torch.where(
                    active.squeeze(-1), convolved, torch.zeros_like(convolved)
                )
            outputs.append(convolved)
        return torch.stack(outputs, dim=1), cache


class KDAAttention(nn.Module):
    """Torch reference for Moonshot's Kimi Delta Attention recurrence."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        conv_size: int = 4,
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

        self.q_proj = nn.Linear(hidden_size, projection_size, bias=False, **factory)
        self.k_proj = nn.Linear(hidden_size, projection_size, bias=False, **factory)
        self.v_proj = nn.Linear(hidden_size, projection_size, bias=False, **factory)
        self.q_conv1d = DepthwiseShortConv(
            projection_size, conv_size, device=device, dtype=dtype
        )
        self.k_conv1d = DepthwiseShortConv(
            projection_size, conv_size, device=device, dtype=dtype
        )
        self.v_conv1d = DepthwiseShortConv(
            projection_size, conv_size, device=device, dtype=dtype
        )
        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(num_heads, device=device, dtype=torch.float32).uniform_(1, 16)
            ).view(1, 1, num_heads, 1)
        )
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **factory)
        self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False, **factory)
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, device=device, dtype=torch.float32)
        )
        nn.init.zeros_(self.dt_bias)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False, **factory)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **factory)
        self.g_b_proj = nn.Linear(head_dim, projection_size, bias=False, **factory)
        self.o_norm = RMSGatedNorm(
            head_dim, eps=rms_norm_eps, device=device, dtype=dtype
        )
        self.o_proj = nn.Linear(projection_size, hidden_size, bias=False, **factory)

    def _decay_gate(self, raw_gate: torch.Tensor) -> torch.Tensor:
        biased = raw_gate.float() + self.dt_bias.float().view(
            self.num_heads, self.head_dim
        )
        rate = self.A_log.float().view(1, 1, self.num_heads, 1).exp()
        return -rate * F.softplus(biased)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: KDALayerState | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KDALayerState]:
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("KDA attention_mask must have shape [batch, sequence]")
        batch, sequence, _ = hidden_states.shape
        if attention_mask is not None and tuple(attention_mask.shape) != (batch, sequence):
            raise ValueError("KDA attention_mask does not match the input shape")

        q, q_state = self.q_conv1d(
            self.q_proj(hidden_states),
            state.q_conv if state is not None else None,
            attention_mask,
        )
        k, k_state = self.k_conv1d(
            self.k_proj(hidden_states),
            state.k_conv if state is not None else None,
            attention_mask,
        )
        v, v_state = self.v_conv1d(
            self.v_proj(hidden_states),
            state.v_conv if state is not None else None,
            attention_mask,
        )

        raw_decay = self.f_b_proj(self.f_a_proj(hidden_states)).view(
            batch, sequence, self.num_heads, self.head_dim
        )
        decay = self._decay_gate(raw_decay)
        beta = torch.sigmoid(self.b_proj(hidden_states).float())
        q = q.view(batch, sequence, self.num_heads, self.head_dim).float()
        k = k.view(batch, sequence, self.num_heads, self.head_dim).float()
        v = v.view(batch, sequence, self.num_heads, self.head_dim).float()
        q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)

        recurrent_shape = (
            batch,
            self.num_heads,
            self.head_dim,
            self.head_dim,
        )
        if state is None:
            recurrent = torch.zeros(
                recurrent_shape,
                dtype=torch.float32,
                device=hidden_states.device,
            )
        else:
            recurrent = state.recurrent.float()
        if tuple(recurrent.shape) != recurrent_shape:
            raise ValueError("KDA recurrent state has an invalid shape")

        outputs: list[torch.Tensor] = []
        scale = self.head_dim**-0.5
        for token_index in range(sequence):
            q_token = q[:, token_index]
            k_token = k[:, token_index]
            v_token = v[:, token_index]
            decay_token = decay[:, token_index]
            beta_token = beta[:, token_index]

            candidate = recurrent * torch.exp(decay_token).unsqueeze(-1)
            prediction = (candidate * k_token.unsqueeze(-1)).sum(dim=-2)
            delta = v_token - prediction
            candidate = candidate + (
                beta_token.unsqueeze(-1).unsqueeze(-1)
                * k_token.unsqueeze(-1)
                * delta.unsqueeze(-2)
            )
            output = (candidate * (q_token * scale).unsqueeze(-1)).sum(dim=-2)
            if attention_mask is not None:
                active = attention_mask[:, token_index].bool().view(batch, 1, 1)
                recurrent = torch.where(active.unsqueeze(-1), candidate, recurrent)
                output = torch.where(active, output, torch.zeros_like(output))
            else:
                recurrent = candidate
            outputs.append(output)

        output = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        output_gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(
            batch, sequence, self.num_heads, self.head_dim
        )
        output = self.o_norm(output, output_gate)
        output = self.o_proj(output.reshape(batch, sequence, self.projection_size))
        final_state = KDALayerState(q_state, k_state, v_state, recurrent)
        if return_state:
            return output, final_state
        return output


class MLAAttention(nn.Module):
    """Direct-query MLA with Moonshot's asserted no-position-embedding path."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        *,
        rms_norm_eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_key_value_heads != num_heads:
            raise ValueError("the real Kimi-Linear checkpoint uses equal query and KV heads")
        factory = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.scaling = self.q_head_dim**-0.5

        self.q_proj = nn.Linear(
            hidden_size, num_heads * self.q_head_dim, bias=False, **factory
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
        if attention_mask is not None:
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (
                batch,
                key_length,
            ):
                raise ValueError("MLA padding mask must cover the complete key sequence")
            allowed = allowed & attention_mask[:, None, None, :].bool()
        additive = torch.zeros(
            batch,
            1,
            query_length,
            key_length,
            dtype=torch.float32,
            device=device,
        )
        additive.masked_fill_(~allowed, torch.finfo(torch.float32).min)
        return additive

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: MLALayerState | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MLALayerState]:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, sequence, self.num_heads, self.q_head_dim
        )
        query = query.transpose(1, 2)
        query_pass, query_rotary = torch.split(
            query, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        current = self.kv_a_proj_with_mqa(hidden_states)
        current_latent, current_rotary = torch.split(
            current, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        if state is None:
            latent = current_latent
            rotary = current_rotary
            past_length = 0
        else:
            if state.batch_size != batch:
                raise ValueError("MLA cache batch size does not match the current input")
            if state.compressed_kv.shape[-1] != self.kv_lora_rank:
                raise ValueError("MLA cache has the wrong latent rank")
            if state.rotary_key.shape[-1] != self.qk_rope_head_dim:
                raise ValueError("MLA cache has the wrong rotary width")
            past_length = state.sequence_length
            latent = torch.cat((state.compressed_kv, current_latent), dim=1)
            rotary = torch.cat((state.rotary_key, current_rotary), dim=1)

        key_pass = self.kv_b_proj(self.kv_a_layernorm(latent)).view(
            batch,
            latent.shape[1],
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_pass = key_pass.transpose(1, 2)
        key_pass, value = torch.split(
            key_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        key_rotary = rotary.view(
            batch, 1, latent.shape[1], self.qk_rope_head_dim
        ).expand(*key_pass.shape[:-1], -1)

        query = torch.cat((query_pass, query_rotary), dim=-1)
        key = torch.cat((key_pass, key_rotary), dim=-1)
        mask = self._attention_mask(
            attention_mask,
            batch,
            sequence,
            latent.shape[1],
            past_length,
            hidden_states.device,
        )
        scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * self.scaling
        probabilities = (scores.float() + mask).softmax(dim=-1).to(query.dtype)
        output = torch.einsum("bhqk,bhkd->bhqd", probabilities, value)
        output = output.transpose(1, 2).reshape(
            batch, sequence, self.num_heads * self.v_head_dim
        )
        output = self.o_proj(output)
        final_state = MLALayerState(latent, rotary)
        if return_state:
            return output, final_state
        return output
