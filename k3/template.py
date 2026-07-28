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
from typing import Any, Iterable, Literal, Optional

from .ir import CanonicalRequest, ToolDef
from .reasoning import (
    normalize_kimi_k3_assistant,
    normalize_kimi_k3_tool_result_messages,
)

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

KIMI_K3_TOOL_PROMPT = """\
# Tools
Here are the available tools, described in JSONSchema.

```json
{tools}
```
"""

_PROMPTS = {
    "kimi": KIMI_TOOL_PROMPT,
    "kimi_k2": KIMI_TOOL_PROMPT,
    "kimi_k3": KIMI_K3_TOOL_PROMPT,
    "hermes": HERMES_TOOL_PROMPT,
}

K3_OPEN = "<|open|>"
K3_CLOSE = "<|close|>"
K3_SEP = "<|sep|>"
K3_END_OF_MSG = "<|end_of_msg|>"
K3_IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
_K3_THINKING_EFFORTS = {"low", "high", "max"}


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
    if style == "kimi_k3":
        payload = _k3_deep_sort(native_tools_payload(tools))
        return template.format(tools=_k3_json_compact(payload)).strip()
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


class _K3ImageState:
    def __init__(self, prompts: Optional[list[str]]) -> None:
        self.prompts = prompts
        self.index = 0

    def next(self) -> str:
        if self.prompts is None:
            return K3_IMAGE_PLACEHOLDER
        if self.index >= len(self.prompts):
            raise ValueError("More image placeholders than image prompts.")
        prompt = self.prompts[self.index]
        self.index += 1
        return prompt

    def assert_consumed(self) -> None:
        if self.prompts is not None and self.index != len(self.prompts):
            raise ValueError(
                f"image prompt count {len(self.prompts)} != "
                f"consumed placeholder count {self.index}"
            )


def render_kimi_k3_prompt(
    messages: list[Any],
    tools: Optional[list[dict[str, Any]]] = None,
    *,
    add_generation_prompt: bool = True,
    thinking: bool = True,
    image_prompts: Optional[list[str]] = None,
    **kwargs: Any,
) -> str:
    """Render the reference Kimi K3 XTML prompt byte for byte."""
    messages = normalize_kimi_k3_tool_result_messages(messages)
    tools = _k3_deep_sort(tools)
    image_state = _K3ImageState(image_prompts)
    out: list[str] = []

    if tools:
        out.append(_k3_tool_declare(tools))

    thinking_effort = kwargs.get("thinking_effort")
    if thinking and thinking_effort is not None:
        assert thinking_effort in _K3_THINKING_EFFORTS, (
            f"Unsupported thinking_effort={thinking_effort!r}; "
            f"supported values are {sorted(_K3_THINKING_EFFORTS)}."
        )
    if thinking and thinking_effort in _K3_THINKING_EFFORTS:
        out.append(
            _k3_internal_system(
                "thinking-effort",
                "`thinking_effort` guides on how much to think in your "
                "thinking channel (not including the response channel), "
                "supported values include `low`, `medium`, `high`, and `max`.\n"
                f"Now the system is invoked with `thinking_effort={thinking_effort}`.",
            )
        )

    tool_calls: Any = None
    tool_index = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message["role"]
        if role in ("user", "system") and not (
            role == "system" and message.get("tools")
        ):
            attrs: list[tuple[str, Any]] = [("role", role)]
            if message.get("name"):
                attrs.append(("name", message["name"]))
            out.append(_k3_open("message", attrs))
            out.append(_k3_content(message.get("content"), image_state))
            out.append(_k3_close("message"))
            out.append(K3_END_OF_MSG)
        elif role == "system":
            out.append(_k3_tool_declare(_k3_deep_sort(message["tools"]), dynamic=True))
        elif role == "assistant":
            normalized = normalize_kimi_k3_assistant(message)
            tool_calls = normalized.get("tool_calls")
            tool_index = 0
            attrs = [("role", "assistant")]
            if normalized.get("name"):
                attrs.append(("name", normalized["name"]))
            out.append(_k3_open("message", attrs))
            out.append(_k3_assistant(normalized, image_state, thinking))
            out.append(_k3_close("message"))
            out.append(K3_END_OF_MSG)
        elif role == "tool":
            tool_index += 1
            tool_name = message.get("tool", message.get("name"))
            if tool_name is None and tool_calls is not None and tool_index <= len(tool_calls):
                function = tool_calls[tool_index - 1].get(
                    "function", tool_calls[tool_index - 1]
                )
                tool_name = function["name"]
            if tool_name is None:
                raise ValueError(
                    "Kimi K3 tool messages need a resolvable tool name: carry "
                    "`tool`/`name`, or match a preceding assistant tool_call by order."
                )
            out.append(
                _k3_open(
                    "message",
                    [("role", "tool"), ("tool", tool_name), ("index", tool_index)],
                )
            )
            out.append(_k3_content(message.get("content"), image_state))
            out.append(_k3_close("message"))
            out.append(K3_END_OF_MSG)

    tool_choice = kwargs.get("tool_choice")
    if tool_choice == "required":
        out.append(
            _k3_internal_system(
                "tool-choice",
                "The system is invoked with `tool_choice=required`.\n"
                "You MUST call tools in the next message.",
            )
        )
    elif tool_choice == "none":
        out.append(
            _k3_internal_system(
                "tool-choice",
                "The system is invoked with `tool_choice=none`.\n"
                "You MUST NOT call any tools in the next message.",
            )
        )

    response_format = kwargs.get("response_format")
    response_type = (
        response_format.get("type", response_format)
        if isinstance(response_format, dict)
        else response_format
    )
    if response_type == "json_object":
        out.append(
            _k3_internal_system(
                "response-format",
                "The system is invoked with `response_format=json_object`.\n"
                "Your response must be raw JSON data without markdown code "
                "blocks (```json) or any additional formatting.",
            )
        )
    elif response_type == "json_schema":
        schema = kwargs.get("response_schema")
        if schema is None and isinstance(response_format, dict):
            json_schema = response_format.get("json_schema")
            if isinstance(json_schema, dict):
                schema = json_schema.get("schema", json_schema.get("json_schema", json_schema))
        schema_text = _k3_json_compact(_k3_deep_sort(schema))
        out.append(
            _k3_internal_system(
                "response-format",
                "The system is invoked with `response_format=json_schema`.\n"
                "Your response must be raw JSON data without markdown code "
                "blocks (```json) or any additional formatting.\n"
                "The JSON data must match the following schema:\n"
                f"```json\n{schema_text}\n```",
            )
        )

    if add_generation_prompt:
        out.append(_k3_open("message", [("role", "assistant")]))
        out.append(_k3_open("think" if thinking else "response"))

    image_state.assert_consumed()
    return "".join(out)


def _k3_assistant(
    message: dict[str, Any], image_state: _K3ImageState, thinking: bool
) -> str:
    out: list[str] = []
    if thinking:
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        out.append(_k3_open("think"))
        if reasoning is not None and str(reasoning).strip():
            out.append(_k3_content(reasoning, image_state))
        out.append(_k3_close("think"))

    out.append(_k3_open("response"))
    out.append(_k3_content(message.get("content"), image_state))
    out.append(_k3_close("response"))

    tool_calls = message.get("tool_calls")
    if tool_calls:
        out.append(_k3_open("tools"))
        for index, tool_call in enumerate(tool_calls, start=1):
            function = tool_call.get("function", tool_call)
            out.append(
                _render_kimi_k3_tool_call(
                    function["name"],
                    function.get("arguments", {}),
                    image_state,
                    index=index,
                    json_block=function.get("_xtml_json_block"),
                )
            )
        out.append(_k3_close("tools"))
    return "".join(out)


def render_kimi_k3_tool_call(
    name: str,
    arguments: Any,
    *,
    index: int = 1,
    json_block: Optional[str] = None,
) -> str:
    """Render one canonical K3 ``call`` element."""
    return _render_kimi_k3_tool_call(
        name,
        arguments,
        _K3ImageState(None),
        index=index,
        json_block=json_block,
    )


def _render_kimi_k3_tool_call(
    name: str,
    arguments: Any,
    image_state: _K3ImageState,
    *,
    index: int,
    json_block: Optional[str],
) -> str:
    out = [_k3_open("call", [("tool", name), ("index", index)])]
    if json_block is not None:
        out.append(_k3_open("json", [("type", "object")]))
        out.append(_k3_replace_images(json_block, image_state))
        out.append(_k3_close("json"))
    elif isinstance(arguments, dict):
        for key, value in arguments.items():
            out.append(
                _k3_open(
                    "argument",
                    [("key", key), ("type", _k3_type(value))],
                )
            )
            out.append(_k3_replace_images(_k3_value(value), image_state))
            out.append(_k3_close("argument"))
    out.append(_k3_close("call"))
    return "".join(out)


def _k3_content(content: Any, image_state: _K3ImageState) -> str:
    if isinstance(content, str):
        return _k3_replace_images(content, image_state)
    if content is None:
        return ""
    out: list[str] = []
    for part in content:
        if part["type"] in ("image", "image_url"):
            out.append(image_state.next())
        else:
            out.append(_k3_replace_images(part["text"], image_state))
    return "".join(out)


def _k3_replace_images(text: Any, image_state: _K3ImageState) -> str:
    text = str(text)
    if image_state.prompts is None or K3_IMAGE_PLACEHOLDER not in text:
        return text
    parts = text.split(K3_IMAGE_PLACEHOLDER)
    out: list[str] = []
    for index, part in enumerate(parts):
        out.append(part)
        if index < len(parts) - 1:
            out.append(image_state.next())
    return "".join(out)


def _k3_open(tag: str, attrs: Iterable[tuple[str, Any]] = ()) -> str:
    rendered = [K3_OPEN, tag]
    for key, value in attrs:
        escaped = str(value).replace("&", "&amp;").replace('"', "&quot;")
        rendered.append(f' {key}="{escaped}"')
    rendered.append(K3_SEP)
    return "".join(rendered)


def _k3_close(tag: str) -> str:
    return K3_CLOSE + tag + K3_SEP


def _k3_internal_system(message_type: str, body: str) -> str:
    return (
        _k3_open("message", [("role", "system"), ("type", message_type)])
        + body.strip()
        + _k3_close("message")
        + K3_END_OF_MSG
    )


def _k3_tool_declare(tools: Any, dynamic: bool = False) -> str:
    if dynamic:
        body = (
            "## New Tools Available\n"
            "The system dynamically extends the toolset via lazy-loading.\n"
            "You have access to all existing and extended tools.\n"
            "Here are the specs for the extended tools.\n\n"
            f"```json\n{_k3_json_compact(tools)}\n```"
        )
    else:
        body = (
            "# Tools\n"
            "Here are the available tools, described in JSONSchema.\n\n"
            f"```json\n{_k3_json_compact(tools)}\n```"
        )
    return (
        _k3_open("message", [("role", "system"), ("type", "tool-declare")])
        + body
        + _k3_close("message")
        + K3_END_OF_MSG
    )


def _k3_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    return "array"


def _k3_value(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _k3_json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _k3_deep_sort(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _k3_deep_sort(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_k3_deep_sort(item) for item in value]
    return value


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
    "render_kimi_k3_prompt",
    "render_kimi_k3_tool_call",
    "collapse_messages",
    "KIMI_TOOL_PROMPT",
    "KIMI_K3_TOOL_PROMPT",
    "HERMES_TOOL_PROMPT",
]
