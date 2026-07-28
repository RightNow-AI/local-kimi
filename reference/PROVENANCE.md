# Reference material

`encoding_k3.py` and `tokenization_kimi.py` are unmodified copies from
`moonshotai/Kimi-K3` on HuggingFace. `kimi_k2_thinking_chat_template.jinja` is
from `moonshotai/Kimi-K2-Thinking`.

They are vendored because `tests/test_k3_oracle.py` executes the real encoder as
a golden oracle: our renderer is diffed against Moonshot's own implementation
byte for byte, which is the only way to be certain the prompt format is right.
They are reference input to the test suite, never imported by `k3/` at runtime.

Licensed under the terms published with those repositories. Before distributing
this project, confirm redistribution is permitted or fetch these at test time
instead of vendoring them.
