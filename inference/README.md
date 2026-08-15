# Lumi Inference (Native HIP)

Native HIP inference for Lumi Mamba-3 models on AMD Radeon AI PRO R9700 (gfx1201 / RDNA4). No ML framework — direct kernel calls, safetensors weight loading, zero-copy device buffers.

**Status:** Under construction. Kernel layer and host interface in progress. See [METAL_TO_HIP.md](METAL_TO_HIP.md) for the porting guide and staged implementation plan.

## Quick Reference

**Build:**
```bash
cargo build --release -p lumi-infer --features hip
```

**Generate:**
```bash
./target/release/lumi-infer generate \
  -m model.safetensors \
  -c config.json \
  -t tokenizer.json \
  -p "Once upon a time"
```

**Smoke test:**
```bash
./target/release/lumi-infer smoke
```

## How it Relates

- **[archive/metal-inference/](../archive/metal-inference/)** — Retired Candle+Metal implementation for Apple Silicon (M4 Pro). Kept for reference; not maintained.
- **[METAL_TO_HIP.md](METAL_TO_HIP.md)** — Detailed porting guide: architectural decisions, kernel traps, staged verification gates, and performance notes. Start here if contributing.

## Architecture

Mirrors the training binary's forward pass: device buffer management (`GpuBuf`), kernel launchers from `lumi-kernels`, safetensors weight loading. Single-token inference via persistent per-layer SSM state (`h`, `prev_bx`, cumulative DD-RoPE angles). O(1) memory per token — no growing KV cache even in hybrid Mamba+Attention configs.

## Weight Export

Training checkpoints must be converted to safetensors:

```bash
python3 scripts/export_native_safetensors.py \
  checkpoints/step-XXXXX --output model.safetensors
```

See `scripts/export_native_safetensors.py` for format details.

## CLI Subcommands

### `generate`

**Usage:** `lumi-infer generate -m MODEL -c CONFIG -t TOKENIZER -p PROMPT [options]`

| Flag | Short | Required | Default | Description |
|------|-------|----------|---------|-------------|
| `--model` | `-m` | yes | — | Path to safetensors weights |
| `--config` | `-c` | yes | — | Path to model config JSON |
| `--tokenizer` | `-t` | no | — | Path to tokenizer JSON (omit for byte-level) |
| `--prompt` | `-p` | yes | — | Input text |
| `--max-tokens` | | no | 200 | Tokens to generate |
| `--temperature` | | no | 0.8 | Sampling temp (≤0 for greedy) |

### `smoke`

Quick kernel test with minimal overhead. No model weights loaded.

**Usage:** `lumi-infer smoke`

## Dependencies

- **lumi-kernels** — Shared kernel library (`kernels/` at workspace root). Built with `--features hip` for AMD HIP backend.
- **hipcc / rocm-developer-tools** — HIP compiler and runtime.
- **safetensors** — Weight format and loader.
