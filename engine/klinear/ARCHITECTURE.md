# Kimi-Linear engine notes

## Layer numbering conclusion

Moonshot's remote `KimiLinearConfig.is_kda_layer` checks `(layer_idx + 1)`
against `linear_attn_config["kda_layers"]`. The decoder constructs zero-based
layer indices and uses MLA when that check is false. The attention lists are
therefore one-based.

The safetensors headers independently confirm this reading. Zero-based layers
3, 7, 11, 15, 19, 23, and 26 contain direct MLA `q_proj`, compressed KV, and
MLA output tensors. The other layers contain KDA projections, short
convolutions, decay tensors, and low-rank output-gate tensors.

Layer 0 has KDA attention but a dense SwiGLU feed-forward block. The public
`layer_kind()` resolver reports it as `dense` because that value describes the
decoder-layer feed-forward form. `attention_kind()` separately and explicitly
reports layer 0 as `kda`. Construction rejects overlapping, missing, or
out-of-range attention classifications.

## Forward semantics

- Dense, routed expert, and shared expert MLPs use `silu(gate) * up` followed
  by the down projection.
- The router selects with sigmoid scores plus the correction bias, gathers the
  uncorrected sigmoid scores, renormalizes the selected scores, and then
  multiplies by `routed_scaling_factor`.
- KDA uses the low-rank `g_a_proj` and `g_b_proj` output gate. Its recurrent
  state is `[batch, heads, key_dim, value_dim]` and does not grow with sequence
  length.
- MLA uses the direct query projection. Moonshot's code asserts
  `mla_use_nope`, then concatenates the stored query and key slices without a
  rotary transform. The engine follows that path.

## Real checkpoint loading

`KLinearModel.from_directory()` reads the Hugging Face safetensors index and
all local shard headers, validates the complete 20,493-tensor name, shape, and
dtype contract, and loads model-level plus non-expert tensors onto the target
device. Routed expert matrices are read only when selected and may be retained
in a bounded device-resident LRU cache. This avoids requiring the complete
98,245,528,576-byte BF16 checkpoint to reside on one H100 at once.

The implementation is intentionally direct PyTorch. It is a correctness path,
not a measured throughput claim.
