# Lumi Model

Heavily based on both existing and described Mamba-3 architecture.

This is a work in progress, this project was originally created to accomplish 2 goals:
- Training can be run on a reasonably priced GPU for an individual (RunPod with the $1.39/hr A100)
- Improve inference from what is available today on M4+ Pro series Apple devices

Cargo workspace containing both the training and inference crates for Lumi, a Mamba-3 language model.

```
model/
├── Cargo.toml        # Workspace manifest
├── training/         # CUDA training binary: lumi
│   ├── src/          # Rust training code (~6.5K lines)
│   ├── csrc/         # Custom CUDA kernels (~3.7K lines)
│   ├── tests/        # 60 unit tests
│   ├── build.rs      # CUDA kernel compilation
│   └── Dockerfile    # RunPod deployment
├── inference/        # Metal inference binary: lumi-infer
│   └── src/          # Candle + Metal code (~3.6K lines + Metal shaders)
└── scripts/          # Shared tooling
    └── export_native_safetensors.py  # Convert checkpoint → safetensors
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
| `lumi smoke-test` | Verify CUDA kernels work |
| `lumi smoke-test --all-configs` | Full forward pass for all config variants |

The trainer expects `data/train.bin`, `data/val.bin`, and `tokenizer.json` in the working directory. Checkpoints are saved to `checkpoints/step-XXXXXX/`.

### Key Training Features

- Pure cuBLAS + custom CUDA kernels (zero framework overhead)
- Fused AdamW with gradient clipping (one kernel per parameter)
- WSD learning rate schedule (warmup-stable-decay)
- Sequential coprime-stride sampling for full data coverage
- Native checkpoint format (`.bin` + `.shape` per tensor, `meta.json`, `optimizer.bin`)

### Tests

```bash
cargo test -p lumi    # 60 tests, CPU only (no CUDA required)
```

## Inference

Runs on Apple Silicon (Metal) or CPU fallback. See [inference/README.md](inference/README.md) for full details.

```bash
# Build
cargo build --release -p lumi-inference

# Export checkpoint
python3 ../scripts/export_native_safetensors.py \
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

Export to safetensors for inference via `scripts/export_native_safetensors.py`.

## Configuration Reference

Training is controlled by a JSON config file with two sections: `model` (architecture) and training hyperparameters.

### Model Config

| Field | Default | Description | Constraints |
|-------|---------|-------------|-------------|
| `d_model` | 1024 | Hidden dimension | Must be divisible by `attn_n_heads` |
| `n_layers` | 48 | Total blocks (Mamba + Attention) | |
| `d_state` | 64 | SSM state dimension | Must be even (DD-RoPE needs d_state/2 pairs) |
| `d_conv` | 4 | Causal conv kernel size | |
| `expand` | 2 | Expansion factor (`d_inner = expand * d_model`) | `d_inner` must be divisible by `n_heads` |
| `n_heads` | 64 | SSM heads | Must be divisible by `n_groups` |
| `n_groups` | 8 | Groups for B, C matrices | |
| `chunk_size` | 256 | SSD algorithm chunk size | |
| `vocab_size` | 32000 | Vocabulary size | Ignored when `byte_level: true` (forced to 259) |
| `max_seq_len` | 2048 | Maximum sequence length | |
| `norm_eps` | 1e-5 | RMSNorm epsilon | |
| `byte_level` | false | Byte-level tokenization mode | When true, vocab_size is forced to 259 (pad/bos/eos + 256 bytes). No tokenizer needed. |

#### Attention (Hybrid Mode)

Pure Mamba by default. Set `attention_interval` or `attention_layers` to add GQA attention blocks.

| Field | Default | Description |
|-------|---------|-------------|
| `attention_interval` | 0 | Insert attention every N layers (0 = pure Mamba, 8 = every 8th layer) |
| `attention_layers` | [] | Explicit attention layer indices (0-indexed). Overrides `attention_interval` if non-empty. |
| `attn_n_heads` | 16 | Query heads in attention blocks |
| `attn_kv_heads` | 4 | KV heads for GQA (query:kv ratio = 4:1) |
| `attn_mlp_expand` | 4 | SwiGLU MLP expand factor in attention blocks |
| `attn_window_sizes` | [] | Sliding window per attention layer (0 or absent = full causal) |

### Training Config

| Field | Default | Description |
|-------|---------|-------------|
| `learning_rate` | 3e-4 | Peak learning rate |
| `min_lr` | 1e-4 | LR floor (33% of peak) |
| `warmup_steps` | 2000 | Linear warmup steps |
| `max_steps` | 15000 | Total training steps |
| `batch_size` | 32 | Per-GPU micro batch size |
| `gradient_accumulation` | 4 | Gradient accumulation steps |
| `weight_decay` | 0.1 | AdamW weight decay |
| `lr_schedule` | "wsd" | LR schedule: `"wsd"` (warmup-stable-decay) or `"cosine"` |
| `decay_fraction` | 0.2 | WSD: fraction of steps in decay phase |
| `warmup_offset` | 0 | Offset warmup start step (for re-warm on checkpoint resume) |
| `checkpoint_interval` | 500 | Steps between checkpoint saves |
| `eval_interval` | 200 | Steps between validation runs |
| `sample_interval` | 1000 | Steps between sample generation |
| `seed` | 42 | RNG seed |

### Example Configs

**Pure Mamba (default):**
```json
{
  "model": {
    "d_model": 1024, "n_layers": 48, "d_state": 64, "d_conv": 4,
    "expand": 2, "n_heads": 64, "n_groups": 8, "chunk_size": 256,
    "vocab_size": 32000, "max_seq_len": 2048, "norm_eps": 1e-5
  },
  "learning_rate": 3e-4, "min_lr": 1e-4,
  "warmup_steps": 2000, "max_steps": 15000,
  "batch_size": 32, "gradient_accumulation": 4,
  "weight_decay": 0.1, "checkpoint_interval": 500,
  "eval_interval": 200, "sample_interval": 1000, "seed": 42
}
```

**Hybrid (attention every 8th layer):**
```json
{
  "model": {
    "d_model": 1024, "n_layers": 48, "d_state": 64, "d_conv": 4,
    "expand": 2, "n_heads": 64, "n_groups": 8, "chunk_size": 256,
    "vocab_size": 32000, "max_seq_len": 2048, "norm_eps": 1e-5,
    "attention_interval": 8,
    "attn_n_heads": 16, "attn_kv_heads": 4, "attn_mlp_expand": 4
  },
  "learning_rate": 3e-4, "min_lr": 1e-4,
  "warmup_steps": 2000, "max_steps": 15000,
  "batch_size": 16, "gradient_accumulation": 4,
  "weight_decay": 0.1, "checkpoint_interval": 500,
  "eval_interval": 200, "sample_interval": 1000, "seed": 42
}
```

**Byte-level (no tokenizer):**
```json
{
  "model": {
    "d_model": 1024, "n_layers": 48, "d_state": 64, "d_conv": 4,
    "expand": 2, "n_heads": 64, "n_groups": 8, "chunk_size": 256,
    "vocab_size": 32000, "max_seq_len": 8192, "norm_eps": 1e-5,
    "byte_level": true
  },
  "learning_rate": 3e-4, "min_lr": 1e-4,
  "warmup_steps": 2000, "max_steps": 15000,
  "batch_size": 8, "gradient_accumulation": 4,
  "weight_decay": 0.1, "checkpoint_interval": 500,
  "eval_interval": 200, "sample_interval": 1000, "seed": 42
}
```

Note: In byte-level mode, `vocab_size` is ignored and forced to 259. Use `lumi preprocess --byte-level` instead of the tokenizer pipeline. Consider increasing `max_seq_len` since bytes are ~4-5x longer than BPE tokens for the same text.

## Tested On
- **Training**: A100 (may require adaptation for other GPUs)
- **Inference**: Mac M4 Pro

---
## Developer Notes

I invite all who feel they have something to contribute to do so, especially if it broadens the hardware compatibility for inference or training (also, happy to merge a simple readme update to say "I ran it fine on _blank_ GPU/machine").

On a personal note, this is the first from-scratch project I've worked on dealing with LLM architecture, native kernels, or Rust. This codebase has been helped a lot by AI friends. I've tried my best to keep it clean, but I'm still learning at the same time, so there may be "dumb" or unnecessary code artifacts. I greatly appreciate any and all constructive criticism or pointers. The main purpose of this project is a fun learning experience outside my normal area of development work.

