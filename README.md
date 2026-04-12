# Lumi Model

Cargo workspace containing both the training and inference crates for Lumi, a Mamba-3 language model.

```
model/
├── Cargo.toml        # Workspace manifest
├── training/         # CUDA training binary: lumi
│   ├── src/          # Rust training code (~6.5K lines)
│   ├── csrc/         # Custom CUDA kernels (~3.7K lines)
│   ├── tests/        # 30 unit tests
│   ├── build.rs      # CUDA kernel compilation
│   └── Dockerfile    # RunPod deployment
└── inference/        # Metal inference binary: lumi-infer
    └── src/          # Candle + Metal code (~3.6K lines + Metal shaders)
```

## Architecture

Mamba-3 selective state space model with these key features:

- **Data-dependent A**: projected from input, computed as `-softplus(dd_A)` clamped to `[-1e6, -1e-4]`
- **Data-dependent lambda**: per-head trapezoidal mixing via sigmoid
- **DD-RoPE**: data-dependent rotary position embeddings on SSM state dimensions
- **BCNorm**: RMSNorm on B/C projections with learnable bias (ones-init)
- **Trapezoidal discretization**: `h[t] = a_bar * h[t-1] + (1-lambda) * a_bar * prev_bx + lambda * bx`
- **Learned h_init**: per-head initial hidden state
- **Optional GQA attention**: 16 query / 4 KV heads with SwiGLU MLP at configurable interval

| Config | d_model | Layers | d_state | n_heads | Params |
|--------|---------|--------|---------|---------|--------|
| Base | 1024 | 48 | 64 | 64 | ~385M |
| Doubled | 1024 | 96 | 64 | 64 | ~970M |

## Training

Requires NVIDIA GPU with CUDA 12+ and Rust toolchain.

```bash
# Build (from model/training/ or via workspace)
cargo build --release -p lumi --features cuda

# Or from the training directory:
cd training && cargo build --release --features cuda
```

### Subcommands

| Command | Description |
|---------|-------------|
| `lumi train --config config.json` | Train model (auto-resumes from latest checkpoint) |
| `lumi preprocess --input corpus.txt --output train.bin` | Tokenize text to binary |
| `lumi train-tokenizer --input corpus.txt --output tokenizer.json` | Train BPE tokenizer |
| `lumi evaluate --config config.json` | Run eval suite (legacy Burn path) |
| `lumi smoke-test` | Verify CUDA kernels work |

The trainer expects `data/train.bin`, `data/val.bin`, and `tokenizer.json` in the working directory. Checkpoints are saved to `checkpoints/step-XXXXXX/`.

### Key Training Features

- Pure cuBLAS + custom CUDA kernels (zero framework overhead)
- Fused AdamW with gradient clipping (one kernel per parameter)
- WSD learning rate schedule (warmup-stable-decay)
- Sequential coprime-stride sampling for full data coverage
- Native checkpoint format (`.bin` + `.shape` per tensor, `meta.json`, `optimizer.bin`)

### Tests

```bash
cargo test -p lumi    # 30 tests, CPU only (no CUDA required)
```

## Inference

Runs on Apple Silicon (Metal) or CPU fallback. See [inference/README.md](inference/README.md) for full details.

```bash
# Build
cargo build --release -p lumi-inference

# Export checkpoint
python3 ../../scripts/inference/export_native_safetensors.py \
  checkpoints/step-013000 --output inference/model.safetensors

# Generate
./target/release/lumi-infer generate \
  -m inference/model.safetensors \
  -c inference/config.json \
  -t ../../tokenizer.json \
  -p "Once upon a time"
```

### Inference Performance (M4 Pro, fp32)

| Model | Speed |
|-------|-------|
| 48-layer (385M) | ~79 tok/s |
| 96-layer (970M) | ~45 tok/s |

## Checkpoint Format

Training saves native binary checkpoints:
```
checkpoints/step-013000/
├── meta.json                          # Step, loss, epoch, sample_pos, tokens_seen
├── optimizer.bin                      # Concatenated Adam m/v state
├── item_embedding_weight.bin + .shape
├── item_blocks_0_in_proj_weight.bin + .shape
├── item_blocks_0_out_proj_weight.bin + .shape
└── ...                                # One .bin + .shape per tensor
```

Export to safetensors for inference via `scripts/inference/export_native_safetensors.py`.

## Tested on:
Training - A100
NOTE: may require adaptation for other gpus
Inference - Mac M4 Pro
