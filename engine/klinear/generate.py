"""Prefill, decode, and token generation for Kimi-Linear."""

from __future__ import annotations

from collections.abc import Generator
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
    decode_backend: str = "eager"


@dataclass
class KLinearGenerationTail:
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


def generate_tokens(
    model: KLinearModel,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    attention_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Generator[torch.Tensor, None, KLinearGenerationTail]:
    """Yield tokens while keeping the following greedy CUDA step in flight.

    CUDA graph streaming queues one decode replay before yielding its current
    token. A consumer-side device-to-host copy can therefore wait for emission
    without preventing the GPU from starting the following token.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    output = prefill(model, prompt_tokens, attention_mask=attention_mask)
    if (
        max_new_tokens
        and prompt_tokens.device.type == "cuda"
        and temperature == 0
    ):
        try:
            runner = CUDAGraphDecodeRunner(
                model, prompt_tokens, output, max_new_tokens
            )
        except RuntimeError:
            runner = None
        if runner is not None:
            for token_index in range(max_new_tokens):
                runner.graph.replay()
                yield runner.generated_with_tail[:, token_index]
            return KLinearGenerationTail(
                runner.graph_state.with_tokens_seen(
                    output.state.tokens_seen + max_new_tokens
                ),
                runner.graph_output.logits,
            )
    state = (
        output.state.reserve_decode_capacity(max_new_tokens)
        if max_new_tokens
        else output.state
    )
    for _ in range(max_new_tokens):
        next_token = sample_logits(
            output.logits[:, -1],
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        yield next_token
        output = decode(model, next_token.unsqueeze(1), state)
        state = output.state
    return KLinearGenerationTail(output.state, output.logits)


def _eager_generate_from_prefill(
    model: KLinearModel,
    prompt_tokens: torch.Tensor,
    output: KLinearModelOutput,
    max_new_tokens: int,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> KLinearGenerationOutput:
    if max_new_tokens == 0:
        generated_ids = prompt_tokens.new_empty(prompt_tokens.shape[0], 0)
        return KLinearGenerationOutput(
            prompt_tokens,
            generated_ids,
            output.state,
            output.logits,
            "eager",
        )
    state = output.state.reserve_decode_capacity(max_new_tokens)
    generated_ids = prompt_tokens.new_empty(
        prompt_tokens.shape[0], max_new_tokens
    )
    for token_index in range(max_new_tokens):
        next_token = sample_logits(
            output.logits[:, -1],
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        generated_ids[:, token_index].copy_(next_token)
        output = decode(model, next_token.unsqueeze(1), state)
        state = output.state
    return KLinearGenerationOutput(
        torch.cat((prompt_tokens, generated_ids), dim=1),
        generated_ids,
        state,
        output.logits,
        "eager-static",
    )


class CUDAGraphDecodeRunner:
    """Reusable fixed-shape greedy decode graph for one prefetched prompt."""

    def __init__(
        self,
        model: KLinearModel,
        prompt_tokens: torch.Tensor,
        output: KLinearModelOutput,
        max_new_tokens: int,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("CUDA graph decode requires at least one token")
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.prefill_output = output
        self.max_new_tokens = max_new_tokens
        first_token = output.logits[:, -1].argmax(dim=-1)
        pristine = output.state.reserve_decode_capacity(max_new_tokens)

        # Compile lazy Triton kernels and CUDA library paths before capture.
        warm_state = pristine.clone_static()
        decode(model, first_token.unsqueeze(1), warm_state)
        torch.cuda.synchronize(prompt_tokens.device)

        self.graph_state = pristine.clone_static()
        self.graph_snapshot = self.graph_state.clone_static()
        self.graph_input = first_token.unsqueeze(1).clone()
        self.first_token = first_token
        self.generated_with_tail = prompt_tokens.new_empty(
            prompt_tokens.shape[0], max_new_tokens + 1
        )
        self.generated_with_tail[:, :1].copy_(self.graph_input)
        self.write_position = torch.ones(
            (), dtype=torch.long, device=prompt_tokens.device
        )
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream(device=prompt_tokens.device)
        self.capture_stream.wait_stream(torch.cuda.current_stream(prompt_tokens.device))
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self.graph_output = decode(model, self.graph_input, self.graph_state)
            next_token = self.graph_output.logits[:, -1].argmax(dim=-1)
            self.generated_with_tail.index_copy_(
                1, self.write_position.reshape(1), next_token.unsqueeze(1)
            )
            self.graph_input.copy_(next_token.unsqueeze(1))
            self.write_position.add_(1)
        torch.cuda.current_stream(prompt_tokens.device).wait_stream(
            self.capture_stream
        )

        self.reset()

    def reset(self) -> None:
        """Restore captured buffers to the post-prefill state."""
        self.graph_state.copy_from_(self.graph_snapshot)
        self.graph_input.copy_(self.first_token.unsqueeze(1))
        self.generated_with_tail[:, :1].copy_(self.graph_input)
        self.write_position.fill_(1)

    def replay(self) -> None:
        for _ in range(self.max_new_tokens):
            self.graph.replay()

    def result(self) -> KLinearGenerationOutput:
        generated_ids = self.generated_with_tail[:, : self.max_new_tokens]
        final_state = self.graph_state.with_tokens_seen(
            self.prefill_output.state.tokens_seen + self.max_new_tokens
        )
        return KLinearGenerationOutput(
            torch.cat((self.prompt_tokens, generated_ids), dim=1),
            generated_ids,
            final_state,
            self.graph_output.logits,
            "cuda-graph",
        )

    def run(self) -> KLinearGenerationOutput:
        self.reset()
        self.replay()
        return self.result()


def _graph_generate_from_prefill(
    model: KLinearModel,
    prompt_tokens: torch.Tensor,
    output: KLinearModelOutput,
    max_new_tokens: int,
) -> KLinearGenerationOutput:
    runner = CUDAGraphDecodeRunner(
        model, prompt_tokens, output, max_new_tokens
    )
    runner.replay()
    return runner.result()


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
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    output = prefill(model, prompt_tokens, attention_mask=attention_mask)
    if (
        max_new_tokens
        and prompt_tokens.device.type == "cuda"
        and temperature == 0
    ):
        try:
            return _graph_generate_from_prefill(
                model, prompt_tokens, output, max_new_tokens
            )
        except RuntimeError:
            # Unsupported capture remains correct and still uses fixed caches.
            pass
    return _eager_generate_from_prefill(
        model,
        prompt_tokens,
        output,
        max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )

