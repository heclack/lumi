# Lumi Inference

Candle + Metal inference binary for Lumi models on Apple Silicon. O(1) memory per token via fixed SSM state (no growing KV cache for Mamba blocks).

## Quick Start

```bash
# Build
cd model && cargo build --release -p lumi-inference

# Generate text
lumi-infer generate \
  -m model.safetensors \
  -c config.json \
  -t tokenizer.json \
  -p "Once upon a time" \
  --max-tokens 200 \
  --temperature 0.8

# Run evaluation (perplexity + MC benchmarks)
lumi-infer evaluate \
  -m model.safetensors \
  -c config.json \
  -t tokenizer.json \
  --val-data data/val.bin \
  --data-dir data
```

Use `--cpu` (global flag, before subcommand) to force CPU instead of Metal GPU.

## Exporting from Training Checkpoints

Native training checkpoints must be converted to safetensors first:

```bash
python3 scripts/export_native_safetensors.py <checkpoint_dir> --output model.safetensors
```

The export script auto-transposes linear weights (`[in, out]` -> `[out, in]`) and reads block count from the checkpoint directory. Works for both 48-layer and 96-layer models.

## Config File

The config JSON must match the model architecture. Example for the 96-layer Phase 2 model:

```json
{
  "d_model": 1024,
  "n_layers": 96,
  "d_state": 64,
  "d_conv": 4,
  "expand": 2,
  "n_heads": 64,
  "n_groups": 8,
  "chunk_size": 256,
  "vocab_size": 32000,
  "max_seq_len": 1024,
  "norm_eps": 1e-5,
  "attention_interval": 0,
  "attn_n_heads": 16,
  "attn_kv_heads": 4,
  "attn_mlp_expand": 4,
  "byte_level": false
}
```

Set `n_layers` to match the exported model (48 for base, 96 for doubled). Set `attention_interval > 0` for hybrid models (e.g. 8 = attention every 8th block).

## CLI Reference

### `generate`

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | required | Path to safetensors weights |
| `--config` | `-c` | required | Path to model config JSON |
| `--tokenizer` | `-t` | optional | Path to tokenizer JSON (omit for byte-level models) |
| `--prompt` | `-p` | required | Input text |
| `--max-tokens` | | 200 | Maximum tokens to generate |
| `--temperature` | | 0.8 | Sampling temperature (<=0 for greedy argmax) |

### `evaluate`

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | required | Path to safetensors weights |
| `--config` | `-c` | required | Path to model config JSON |
| `--tokenizer` | `-t` | optional | Path to tokenizer JSON (omit for byte-level models) |
| `--val-data` | | `data/val.bin` | Validation data for perplexity |
| `--data-dir` | | `data` | Directory with benchmark `.txt` files (ARC-Easy, BoolQ, WinoGrande) |

## Architecture

Each token is processed through `forward_step()` with persistent per-layer state:

- **Mamba blocks**: O(1) SSM state (`h`, `prev_bx`, cumulative DD-RoPE angles). Full Mamba-3: data-dependent A, data-dependent lambda, DD-RoPE, trapezoidal discretization, BCNorm.
- **Attention blocks** (hybrid mode): KV cache grows with sequence. Optional sliding window with FIFO truncation.

Seven fused Metal kernels minimize dispatch overhead — the entire single-token generation path is 5 custom dispatches per Mamba block + 1 for logit projection:

| Kernel | What it replaces | Dispatches saved |
|--------|-----------------|-----------------|
| **RmsNormPipeline** | RMSNorm (sqr, mean, rsqrt, mul) | 3/block |
| **GemvPipeline** | in_proj Linear (unsqueeze + MPS matmul + squeeze) | ~2/block |
| **PreSsmPipeline** | ~35 ops (SiLU, BCNorm, softplus, sigmoid, group expansion, RoPE prep) | 34/block |
| **SsmStepPipeline** | DD-RoPE + SSM scan + output contraction + Z-gate | already fused |
| **GemvResidualPipeline** | out_proj Linear + residual add | ~3/block |
| **SsmWindowPipeline** | Per-token SSM scan in eval (loops over full sequence in 1 dispatch) | N/window |
| **GemvPipeline** (logit) | Final embedding^T projection | ~2/model |

Performance on M4 Pro (Apple Silicon, fp32):
- **48-layer (385M)**: ~79 tok/s generation
- **96-layer (970M)**: ~45 tok/s generation

Falls back to CPU (Candle ops) if Metal is unavailable.

## Weight Compatibility

Loads safetensors exported via `scripts/export_native_safetensors.py`. Handles missing parameters gracefully:
- `b_bias` / `c_bias`: defaults to ones (backward compat with older checkpoints)
- `h_init`: defaults to zeros
