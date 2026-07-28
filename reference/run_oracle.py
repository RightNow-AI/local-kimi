"""Run the REAL Kimi K3 prompt encoder locally. No GPU, no weights, no deps."""
import importlib.util, inspect, json, pathlib, sys

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("encoding_k3", HERE / "k3_encoding_k3.py")
enc = importlib.util.module_from_spec(spec)
sys.modules["encoding_k3"] = enc
spec.loader.exec_module(enc)
print("imported encoding_k3 with ZERO external deps\n")

print("build_chat_segments signature:")
print("  ", inspect.signature(enc.build_chat_segments), "\n")

print("EncodeSegment fields:", [f.name for f in __import__("dataclasses").fields(enc.EncodeSegment)], "\n")

CONV = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is the weather in Beijing?"},
    {
        "role": "assistant",
        "reasoning_content": "The user wants weather. I should call get_weather.",
        "content": "",
        "tool_calls": [{
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":  "Beijing", "units":"c"}'},
        }],
    },
    {"role": "tool", "tool_call_id": "call_abc123", "content": "22C, sunny"},
    {"role": "assistant", "reasoning_content": "Got it.", "content": "22C and sunny."},
]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]

kw = {}
params = inspect.signature(enc.build_chat_segments).parameters
for name, val in (("tools", TOOLS), ("thinking", True), ("add_generation_prompt", True)):
    if name in params:
        kw[name] = val
print("calling build_chat_segments(conversation, **%s)\n" % list(kw))

segs = enc.build_chat_segments(CONV, **kw)
rendered = "".join(getattr(s, "text", "") or "" for s in segs)

print("=" * 70)
print("REAL K3 PROMPT (%d segments, %d chars)" % (len(segs), len(rendered)))
print("=" * 70)
print(rendered)
print("=" * 70)

out = HERE / "k3_golden_prompt.txt"
out.write_text(rendered, encoding="utf-8")
print("\nsaved ->", out)

print("\n=== does OUR parser's K2 format appear anywhere? ===")
for tok in ("<|tool_calls_section_begin|>", "<|tool_call_begin|>", "<|tool_call_argument_begin|>",
            "<|tool_call_end|>", "functions.get_weather"):
    print(f"  {'FOUND' if tok in rendered else 'ABSENT':7s} {tok}")

print("\n=== is the raw argument JSON preserved byte-exactly? ===")
raw = '{"city":  "Beijing", "units":"c"}'
print(f"  raw model arguments : {raw!r}")
print(f"  present verbatim    : {raw in rendered}")
