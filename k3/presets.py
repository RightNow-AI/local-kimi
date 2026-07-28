"""Presets — the whole point of the project.

A preset bundles the six things a client needs to talk to K3 correctly:

    route + dialect        which endpoints, Messages vs Chat vs Responses
    tool parser + emitter  how calls come out of the model and into the client
    chat template          how messages and tool defs render into the prompt
    reasoning translation   both directions, see k3/reasoning.py
    defaults               reasoning effort, max tokens, streaming shape
    model aliasing         whatever model string the client asks for -> K3

Presets carry a ``status``. ``stable`` means there is recorded client traffic in
``tests/cassettes`` pinning the behaviour. ``provisional`` means the preset is
built from the client's documented wire format but has not been pinned by a
recording yet — it is expected to work, and it is expected to be the thing that
breaks when the client ships an update. Promote a preset by recording real
traffic (``k3 serve --record DIR``) and dropping the cassette in.

``kimi`` and ``kimi_k2`` select the legacy K2 control-token parser when an
older engine needs it. K3 presets default to ``kimi_k3`` so production parses
the XTML shape the model actually emits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from .reasoning import ReasoningPolicy
from .template import TemplateConfig

Dialect = Literal["anthropic_messages", "openai_chat", "openai_responses"]
Status = Literal["stable", "provisional"]

TOOL_PARSER_DESCRIPTIONS = {
    "kimi": "Kimi K2 control-token format (compatibility alias)",
    "kimi_k2": "Kimi K2 control-token format",
    "kimi_k3": "Kimi K3 XTML format",
}


@dataclass(slots=True)
class Defaults:
    max_tokens: int = 8192
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    #: Ask the engine for a usage block on the final streaming chunk.
    stream_include_usage: bool = True
    #: Emit SSE keepalives; Claude Code in particular expects `ping` events.
    stream_keepalive: bool = False
    parallel_tool_calls: Optional[bool] = None


@dataclass(slots=True)
class ModelAliasing:
    """Whatever model string the client asks for resolves to K3."""

    #: Model ids advertised on ``GET /v1/models``.
    advertise: list[str] = field(default_factory=list)
    #: Requests whose model matches one of these route to the small model, if
    #: one is configured (Claude Code uses a cheap model for background work).
    small_patterns: list[str] = field(default_factory=list)
    #: Model ids passed to the engine untouched instead of being rewritten.
    passthrough_patterns: list[str] = field(default_factory=list)

    def is_small(self, model: str) -> bool:
        return any(re.search(p, model, re.I) for p in self.small_patterns)

    def is_passthrough(self, model: str) -> bool:
        return any(re.search(p, model, re.I) for p in self.passthrough_patterns)


@dataclass(slots=True)
class DetectRules:
    """How we recognise this client from the request itself."""

    paths: list[str] = field(default_factory=list)
    #: header name (lowercase) -> regex the value must match
    headers: dict[str, str] = field(default_factory=dict)
    user_agents: list[str] = field(default_factory=list)
    #: top-level body keys whose presence is evidence for this preset
    body_keys: list[str] = field(default_factory=list)
    #: tie-breaker; the generic fallbacks sit at 0
    weight: int = 0


@dataclass(slots=True)
class Setup:
    """The copy-paste block printed at startup."""

    env: list[tuple[str, str]] = field(default_factory=list)
    launch: str = ""
    config_file: Optional[str] = None
    config_body: str = ""

    def render_env(self, base_url: str, token: str, model: str, shell: str = "posix") -> list[str]:
        lines = []
        for key, value in self.env:
            value = value.format(base_url=base_url, token=token, model=model)
            if shell == "powershell":
                lines.append(f'$env:{key} = "{value}"')
            elif shell == "cmd":
                lines.append(f"set {key}={value}")
            else:
                lines.append(f"export {key}={value}")
        return lines

    def render_config(self, base_url: str, token: str, model: str) -> str:
        return self.config_body.format(base_url=base_url, token=token, model=model)


@dataclass(slots=True)
class Preset:
    name: str
    title: str
    dialect: Dialect
    routes: list[str]
    tool_parser: str
    reasoning: ReasoningPolicy
    template: TemplateConfig = field(default_factory=TemplateConfig)
    defaults: Defaults = field(default_factory=Defaults)
    models: ModelAliasing = field(default_factory=ModelAliasing)
    detect: DetectRules = field(default_factory=DetectRules)
    setup: Setup = field(default_factory=Setup)
    status: Status = "provisional"
    notes: str = ""

    def describe(self) -> str:
        return (
            f"{self.name} [{self.status}] {self.dialect} "
            f"tools={self.tool_parser} reasoning={self.reasoning.value}"
        )


# --------------------------------------------------------------------------
# the presets
# --------------------------------------------------------------------------


CLAUDE_CODE = Preset(
    name="claude-code",
    title="Claude Code",
    dialect="anthropic_messages",
    routes=["/v1/messages", "/v1/messages/count_tokens", "/v1/models"],
    tool_parser="kimi_k3",
    reasoning=ReasoningPolicy.THINKING_BLOCKS,
    template=TemplateConfig(tool_mode="native", system_mode="merge"),
    defaults=Defaults(
        max_tokens=32000,
        temperature=1.0,
        reasoning_effort="medium",
        stream_keepalive=True,
    ),
    models=ModelAliasing(
        advertise=[
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-1-20250805",
            "claude-3-5-haiku-20241022",
        ],
        small_patterns=[r"haiku"],
    ),
    detect=DetectRules(
        paths=["/v1/messages", "/v1/models"],
        headers={"anthropic-version": r".+"},
        user_agents=[r"claude-cli/"],
        body_keys=["system", "max_tokens"],
        weight=30,
    ),
    setup=Setup(
        env=[
            ("ANTHROPIC_BASE_URL", "{base_url}"),
            ("ANTHROPIC_AUTH_TOKEN", "{token}"),
            ("ANTHROPIC_MODEL", "{model}"),
            ("ANTHROPIC_SMALL_FAST_MODEL", "{model}"),
        ],
        launch="claude",
    ),
    status="stable",
    notes=(
        "Reasoning rides in `thinking` blocks; the signature carries the exact "
        "reasoning_content bytes so K3 gets them back verbatim next turn."
    ),
)


OPENAI = Preset(
    name="openai",
    title="Generic OpenAI client",
    dialect="openai_chat",
    routes=["/v1/chat/completions", "/v1/models"],
    tool_parser="kimi_k3",
    reasoning=ReasoningPolicy.STRIP,
    template=TemplateConfig(tool_mode="native"),
    defaults=Defaults(max_tokens=8192),
    models=ModelAliasing(advertise=["k3", "gpt-4o", "gpt-4o-mini"]),
    detect=DetectRules(paths=["/v1/chat/completions", "/v1/models"], weight=0),
    setup=Setup(
        env=[("OPENAI_BASE_URL", "{base_url}/v1"), ("OPENAI_API_KEY", "{token}")],
        launch="",
    ),
    status="stable",
    notes=(
        "Reasoning is stripped from the response because the client would choke "
        "on it, then recovered by fingerprint on the next turn so K3 still sees "
        "its own chain of thought."
    ),
)


CODEX = Preset(
    name="codex",
    title="OpenAI Codex CLI",
    dialect="openai_responses",
    routes=["/v1/responses", "/v1/models"],
    tool_parser="kimi_k3",
    reasoning=ReasoningPolicy.RESPONSES_ITEM,
    template=TemplateConfig(tool_mode="native"),
    defaults=Defaults(max_tokens=32000, reasoning_effort="medium"),
    models=ModelAliasing(advertise=["k3", "gpt-5-codex", "o4-mini"]),
    detect=DetectRules(
        paths=["/v1/responses", "/v1/models"],
        user_agents=[r"codex[_-]cli", r"codex/"],
        body_keys=["input", "instructions"],
        weight=30,
    ),
    setup=Setup(
        env=[("K3_API_KEY", "{token}")],
        launch="codex",
        config_file="~/.codex/config.toml",
        config_body="""\
model = "{model}"
model_provider = "k3"

[model_providers.k3]
name = "k3"
base_url = "{base_url}/v1"
wire_api = "responses"
env_key = "K3_API_KEY"
""",
    ),
    status="stable",
    notes=(
        "Reasoning rides in `reasoning` output items; encrypted_content carries "
        "the exact bytes back. Codex echoes reasoning items it did not author, "
        "so the ledger is the primary recovery path here."
    ),
)


KIMI_CODE = Preset(
    name="kimi-code",
    title="Kimi Code / Kimi CLI",
    dialect="openai_chat",
    routes=["/v1/chat/completions", "/v1/models"],
    tool_parser="kimi_k3",
    reasoning=ReasoningPolicy.REASONING_CONTENT,
    template=TemplateConfig(tool_mode="native"),
    defaults=Defaults(max_tokens=32000, temperature=0.6, reasoning_effort="medium"),
    models=ModelAliasing(
        advertise=["k3", "kimi-k2-thinking", "kimi-latest"],
        passthrough_patterns=[r"^kimi-"],
    ),
    detect=DetectRules(
        paths=["/v1/chat/completions", "/v1/models"],
        user_agents=[r"kimi[_-]?(code|cli)"],
        headers={"x-msh-client": r".+"},
        weight=20,
    ),
    setup=Setup(
        env=[
            ("MOONSHOT_BASE_URL", "{base_url}/v1"),
            ("MOONSHOT_API_KEY", "{token}"),
        ],
        launch="kimi",
    ),
    notes=(
        "The correctness reference: reasoning_content goes out and comes back "
        "untouched, so this preset is the shape K3 was trained against. When "
        "another preset disagrees with this one, the other preset is wrong."
    ),
)


CLINE = Preset(
    name="cline",
    title="Cline",
    dialect="openai_chat",
    routes=["/v1/chat/completions", "/v1/models"],
    tool_parser="kimi_k3",
    reasoning=ReasoningPolicy.REASONING_CONTENT,
    template=TemplateConfig(tool_mode="native"),
    defaults=Defaults(max_tokens=16384, temperature=0.0),
    models=ModelAliasing(advertise=["k3"]),
    detect=DetectRules(
        paths=["/v1/chat/completions", "/v1/models"],
        headers={"http-referer": r"cline\.bot", "x-title": r"(?i)cline"},
        user_agents=[r"(?i)cline"],
        weight=20,
    ),
    setup=Setup(
        env=[("OPENAI_BASE_URL", "{base_url}/v1"), ("OPENAI_API_KEY", "{token}")],
        launch="",
    ),
    notes="Cline renders reasoning_content in its own pane, so we pass it through.",
)


OPENCODE = Preset(
    name="opencode",
    title="OpenCode",
    dialect="openai_chat",
    routes=["/v1/chat/completions", "/v1/models"],
    tool_parser="kimi_k3",
    reasoning=ReasoningPolicy.REASONING_CONTENT,
    template=TemplateConfig(tool_mode="native"),
    defaults=Defaults(max_tokens=16384),
    models=ModelAliasing(advertise=["k3"]),
    detect=DetectRules(
        paths=["/v1/chat/completions", "/v1/models"],
        user_agents=[r"(?i)opencode"],
        weight=20,
    ),
    setup=Setup(
        env=[("OPENAI_BASE_URL", "{base_url}/v1"), ("OPENAI_API_KEY", "{token}")],
        launch="opencode",
    ),
)


AIDER = Preset(
    name="aider",
    title="Aider",
    dialect="openai_chat",
    routes=["/v1/chat/completions", "/v1/models"],
    tool_parser="kimi_k3",
    reasoning=ReasoningPolicy.INLINE_TAGS,
    template=TemplateConfig(tool_mode="native"),
    defaults=Defaults(max_tokens=16384, temperature=0.0),
    models=ModelAliasing(advertise=["k3"]),
    detect=DetectRules(
        paths=["/v1/chat/completions", "/v1/models"],
        user_agents=[r"(?i)aider", r"(?i)litellm"],
        weight=10,
    ),
    setup=Setup(
        env=[
            ("OPENAI_API_BASE", "{base_url}/v1"),
            ("OPENAI_API_KEY", "{token}"),
        ],
        launch="aider --model openai/{model}",
    ),
    notes=(
        "Aider has no reasoning channel, so reasoning is wrapped in <think> tags "
        "inside the content where aider's own filter already knows to hide it."
    ),
)


ALL_PRESETS: list[Preset] = [
    CLAUDE_CODE,
    OPENAI,
    CODEX,
    KIMI_CODE,
    CLINE,
    OPENCODE,
    AIDER,
]

_BY_NAME = {p.name: p for p in ALL_PRESETS}

#: Which preset handles a route when detection finds nothing more specific.
FALLBACK_BY_PATH = {
    "/v1/messages": "claude-code",
    "/v1/messages/count_tokens": "claude-code",
    "/v1/responses": "codex",
    "/v1/chat/completions": "openai",
    # Every client polls /v1/models; the shape differs per dialect, so it needs
    # a fallback of its own rather than inheriting whatever ran last.
    "/v1/models": "openai",
}


def get(name: str) -> Preset:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown preset {name!r}; known: {', '.join(sorted(_BY_NAME))}"
        ) from None


def names() -> list[str]:
    return [p.name for p in ALL_PRESETS]


def all_presets() -> list[Preset]:
    return list(ALL_PRESETS)


def validate(presets: Iterable[Preset] = ()) -> list[str]:
    """Structural checks that catch preset bugs at import/CLI time."""
    problems: list[str] = []
    for p in presets or ALL_PRESETS:
        if p.template.tool_mode == "prompted" and p.tool_parser in ("passthrough", "none"):
            problems.append(
                f"{p.name}: prompted tools need a text tool parser, got {p.tool_parser!r}"
            )
        if p.template.tool_mode == "native" and p.tool_parser == "":
            problems.append(f"{p.name}: empty tool parser")
        if p.dialect == "anthropic_messages" and p.reasoning not in (
            ReasoningPolicy.THINKING_BLOCKS,
            ReasoningPolicy.INLINE_TAGS,
            ReasoningPolicy.STRIP,
        ):
            problems.append(
                f"{p.name}: {p.reasoning.value} has no representation in the Messages API"
            )
        if p.dialect == "openai_responses" and p.reasoning not in (
            ReasoningPolicy.RESPONSES_ITEM,
            ReasoningPolicy.STRIP,
        ):
            problems.append(
                f"{p.name}: {p.reasoning.value} has no representation in the Responses API"
            )
        if not p.routes:
            problems.append(f"{p.name}: no routes")
    return problems


__all__ = [
    "Preset",
    "Defaults",
    "ModelAliasing",
    "DetectRules",
    "Setup",
    "Dialect",
    "TOOL_PARSER_DESCRIPTIONS",
    "ALL_PRESETS",
    "FALLBACK_BY_PATH",
    "get",
    "names",
    "all_presets",
    "validate",
]
