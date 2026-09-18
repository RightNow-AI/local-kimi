# OpenAI Python SDK

The `openai` preset uses OpenAI Chat Completions at `/v1/chat/completions`. The route is
also the fallback for an otherwise unrecognised Chat Completions client.

Complete the [quickstart](QUICKSTART.md) first if the model server is not already
running.

Start `k3` against the local llama.cpp server:

```bash
uv run k3 serve --upstream http://127.0.0.1:8000/v1 --model kimi-linear --reasoning-field inline
```

Install the SDK if it is not already present:

```bash
python3 -m pip install --upgrade openai
```

Set the environment used by the SDK:

```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=local
```

Save this as `hello_kimi.py`:

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="kimi-linear",
    messages=[
        {"role": "user", "content": "Reply with exactly: OpenAI client connected"}
    ],
)

print(response.choices[0].message.content)
```

Run one request:

```bash
python3 hello_kimi.py
```

The token value is arbitrary while `k3` is started without `--api-key`.

The generic `openai` preset strips reasoning from the client response because a
plain Chat Completions client may not accept a nonstandard field. While the
server process remains alive, `k3` can recover reasoning on the next turn by a
fingerprint of visible text and tool calls. If the backend emitted no reasoning
channel in the first place, there is nothing separate to recover.
