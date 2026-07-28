"""Chat template: how messages and tool definitions render into K3's prompt.

When the engine was started with its own tool parser, tools go over the wire as
a native ``tools`` array and this module barely does anything. When they didn't,
we render the tool definitions into the system prompt ourselves and rely on the
matching text parser to read the calls back out. A preset picks which, and the
two halves have to agree — a ``prompted`` template with a ``passthrough`` parser
is a preset bug, and :func:`k3.presets.validate` rejects it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from .ir import CanonicalRequest, ToolDef

ToolMode = Literal["native", "prompted"]
SystemMode = Literal["merge", "separate"]


KIMI_TOOL_PROMPT = """\
# Tools

You have access to the following functions. To call one, emit exactly:

<|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:0\
<|tool_call_argument_begin|>{{"arg": "value"}}<|tool_call_end|>\
<|tool_calls_section_end|>

Emit one section containing every call you want to make. Arguments must be a
single JSON object matching the function's schema. Do not wrap the section in
markdown, and do not explain the call before making it.

{tools}
"""

HERMES_TOOL_PROMPT = """\
# Tools

You may call one or more functions. For each call emit:

<tool_call>{{"name": "FUNCTION_NAME", "arguments": {{"arg": "value"}}}}</tool_call>

{tools}
"""

_PROMPTS = {
    "kimi": KIMI_TOOL_PROMPT,
    "kimi_k2": KIMI_TOOL_PROMPT,
    "hermes": HERMES_TOOL_PROMPT,
}


@dataclass(slots=True)
class TemplateConfig:
    #: ``native`` sends a ``tools`` array; ``prompted`` renders them in-prompt.
    tool_mode: ToolMode = "native"
    #: ``merge`` folds all system blocks into one leading system message.
    system_mode: SystemMode = "merge"
    #: Text prepended to the system prompt for every request under this preset.
    system_prefix: str = ""
    #: Text appended to the system prompt for every request under this preset.
    system_suffix: str = ""
    #: Which prompt to use when ``tool_mode == "prompted"``.
    tool_prompt_style: str = "kimi"
    #: Drop empty assistant turns, which some engines reject.
    drop_empty_assistant: bool = True
    #: Collapse consecutive same-role messages (some templates require strict
    #: alternation).
    collapse_consecutive: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def render_tool_defs(tools: list[ToolDef]) -> str:
    """One JSON schema block per tool, stable ordering, readable in a prompt."""
    blocks = []
    for tool in tools:
        payload = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        blocks.append(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False))
    return "\n\n".join(blocks)


def render_tool_prompt(tools: list[ToolDef], style: str = "kimi") -> str:
    if not tools:
        return ""
    template = _PROMPTS.get(style, KIMI_TOOL_PROMPT)
    return template.format(tools=render_tool_defs(tools))


def build_system_prompt(req: CanonicalRequest, cfg: TemplateConfig) -> Optional[str]:
    """Assemble the single system string the engine should see."""
    chunks: list[str] = []
    if cfg.system_prefix:
        chunks.append(cfg.system_prefix)
    chunks.extend(s for s in req.system if s)
    if cfg.tool_mode == "prompted" and req.tools:
        chunks.append(render_tool_prompt(req.tools, cfg.tool_prompt_style))
    if cfg.system_suffix:
        chunks.append(cfg.system_suffix)
    joined = "\n\n".join(c.strip() for c in chunks if c and c.strip())
    return joined or None


def native_tools_payload(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def collapse_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive same-role text messages, preserving everything else."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if (
            out
            and out[-1].get("role") == msg.get("role")
            and msg.get("role") in ("user", "system")
            and isinstance(out[-1].get("content"), str)
            and isinstance(msg.get("content"), str)
        ):
            out[-1] = dict(out[-1])
            out[-1]["content"] = f"{out[-1]['content']}\n\n{msg['content']}"
        else:
            out.append(msg)
    return out


__all__ = [
    "TemplateConfig",
    "ToolMode",
    "SystemMode",
    "render_tool_defs",
    "render_tool_prompt",
    "build_system_prompt",
    "native_tools_payload",
    "collapse_messages",
    "KIMI_TOOL_PROMPT",
    "HERMES_TOOL_PROMPT",
]
