# Engine serve

`engine.serve` exposes the engine through the three endpoints K3 needs:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Run the CPU-only echo implementation with:

```powershell
python -m engine.serve
```

Point K3 at `http://127.0.0.1:8000/v1` to exercise its real upstream path.

Production engines implement `InferenceEngine.generate`. Each call receives
prompt token IDs and the same sampling controls accepted by
`engine.k3ref.generate`: `max_tokens`, `temperature`, and `top_p`. It yields
sampled token IDs followed by one exact usage record. Closing the iterator must
cancel that request and release its decode resources.

Production tokenizers implement `ChatTokenizer`. This boundary owns the real
chat template and returns a fresh incremental decoder per request. That decoder
must identify reasoning and visible response text separately so the HTTP layer
can emit `reasoning_content` without guessing about model-specific control
tokens.

Generation is serialized by default because the current reference engine has
single-request decode state. A deployment with isolated request state may set
`ServerConfig(serialize_engine=False)`. This changes concurrency only, not the
engine or tokenizer interfaces.
