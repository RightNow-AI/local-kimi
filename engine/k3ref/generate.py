"""Obviously-correct prefill and token-at-a-time generation for Kimi K3."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .model import K3Model, K3ModelOutput
from .state import K3DecodeState


@dataclass
class K3GenerationOutput:
    token_ids: torch.Tensor
    generated_ids: torch.Tensor
    state: K3DecodeState
    final_logits: torch.Tensor


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row, with temperature zero defined as greedy."""

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
            dim=-1,
            descending=True,
        )
        cumulative = sorted_probabilities.cumsum(dim=-1)
        # Keep the first token whose inclusion reaches the requested mass.
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
            dim=-1,
            keepdim=True,
        )
        sampled_sorted = torch.multinomial(
            sorted_probabilities,
            num_samples=1,
            generator=generator,
        )
        return sorted_indices.gather(-1, sampled_sorted).squeeze(-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(-1)


@torch.inference_mode()
def prefill(
    model: K3Model,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    state: K3DecodeState | None = None,
) -> K3ModelOutput:
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("prefill input_ids must have shape [batch, sequence > 0]")
    return model(
        input_ids,
        attention_mask=attention_mask,
        state=state,
    )


@torch.inference_mode()
def decode(
    model: K3Model,
    token_ids: torch.Tensor,
    state: K3DecodeState,
    *,
    attention_mask: torch.Tensor | None = None,
) -> K3ModelOutput:
    if token_ids.ndim != 2 or token_ids.shape[1] != 1:
        raise ValueError("decode consumes exactly one token per batch item")
    return model(
        token_ids,
        attention_mask=attention_mask,
        state=state,
    )


@torch.inference_mode()
def generate(
    model: K3Model,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    attention_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    observer: Callable[[str, int, K3ModelOutput], None] | None = None,
) -> K3GenerationOutput:
    """Prefill once, then feed each sampled token back with the returned state."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    output = prefill(model, prompt_tokens, attention_mask=attention_mask)
    if output.logits is None:
        raise ValueError("generation requires a model with an embedding and LM head")
    if observer is not None:
        observer("prefill", -1, output)

    generated: list[torch.Tensor] = []
    for step in range(max_new_tokens):
        next_token = sample_logits(
            output.logits[:, -1],
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        generated.append(next_token)
        output = decode(model, next_token.unsqueeze(1), output.state)
        if output.logits is None:
            raise RuntimeError("model stopped returning logits during decode")
        if observer is not None:
            observer("decode", step, output)

    if generated:
        generated_ids = torch.stack(generated, dim=1)
        token_ids = torch.cat((prompt_tokens, generated_ids), dim=1)
    else:
        generated_ids = prompt_tokens.new_empty((prompt_tokens.shape[0], 0))
        token_ids = prompt_tokens
    return K3GenerationOutput(
        token_ids,
        generated_ids,
        output.state,
        output.logits,
    )
