"""Prefill, decode, and token generation for Kimi-Linear."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import KLinearModel, KLinearModelOutput
from .state import KLinearDecodeState


@dataclass
class KLinearGenerationOutput:
    token_ids: torch.Tensor
    generated_ids: torch.Tensor
    state: KLinearDecodeState
    final_logits: torch.Tensor


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return logits.argmax(dim=-1)
    probabilities = (logits.float() / temperature).softmax(dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = probabilities.sort(
            dim=-1, descending=True
        )
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
            dim=-1, keepdim=True
        )
        sampled = torch.multinomial(
            sorted_probabilities, num_samples=1, generator=generator
        )
        return sorted_indices.gather(-1, sampled).squeeze(-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)


@torch.inference_mode()
def prefill(
    model: KLinearModel,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    state: KLinearDecodeState | None = None,
) -> KLinearModelOutput:
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("prefill input_ids must have shape [batch, sequence > 0]")
    return model(input_ids, attention_mask=attention_mask, state=state)


@torch.inference_mode()
def decode(
    model: KLinearModel,
    token_ids: torch.Tensor,
    state: KLinearDecodeState,
    *,
    attention_mask: torch.Tensor | None = None,
) -> KLinearModelOutput:
    if token_ids.ndim != 2 or token_ids.shape[1] != 1:
        raise ValueError("decode consumes exactly one token per batch item")
    return model(token_ids, attention_mask=attention_mask, state=state)


@torch.inference_mode()
def generate(
    model: KLinearModel,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    attention_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> KLinearGenerationOutput:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    output = prefill(model, prompt_tokens, attention_mask=attention_mask)
    generated: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        next_token = sample_logits(
            output.logits[:, -1],
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        generated.append(next_token)
        output = decode(model, next_token.unsqueeze(1), output.state)
    if generated:
        generated_ids = torch.stack(generated, dim=1)
        token_ids = torch.cat((prompt_tokens, generated_ids), dim=1)
    else:
        generated_ids = prompt_tokens.new_empty(prompt_tokens.shape[0], 0)
        token_ids = prompt_tokens
    return KLinearGenerationOutput(
        token_ids, generated_ids, output.state, output.logits
    )

