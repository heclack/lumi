/// Weight loading: safetensors file -> per-layer GpuBufs.
///
/// Tensor names and shapes must match `scripts/export_native_safetensors.py`
/// exactly (see also `training/src/native_checkpoint.rs`, which writes the raw
/// binaries the export script reads).
///
/// One layout wrinkle: the export script transposes every 2D `.weight` tensor
/// (except the embedding) from the trainer's native `[in_features,
/// out_features]` layout to `[out_features, in_features]` so Candle's
/// `Linear` (which computes `x @ W^T`) can load it directly. This crate has no
/// Candle and calls `matmul_f32` (plain `A @ B`, not transposed) for `in_proj`
/// and `out_proj`, so we undo that transpose on load — see `load_weight_matrix`.
use anyhow::{bail, Context, Result};
use lumi_kernels::buf::GpuBuf;
use safetensors::{Dtype, SafeTensors};

use crate::config::ModelConfig;

/// Weights for one Mamba block.
pub struct MambaLayerWeights {
    /// `[d_model, in_proj_out]` — native (untransposed) layout for `matmul_f32`.
    pub in_proj: GpuBuf,
    /// `[d_inner, d_model]` — native (untransposed) layout for `matmul_f32`.
    pub out_proj: GpuBuf,
    pub norm_gamma: GpuBuf, // [d_model]
    pub b_gamma: GpuBuf,    // [bc_size]
    pub c_gamma: GpuBuf,    // [bc_size]
    pub b_bias: GpuBuf,     // [bc_size]
    pub c_bias: GpuBuf,     // [bc_size]
    pub d_skip: GpuBuf,     // [n_heads]
    pub dt_bias: GpuBuf,    // [n_heads]
    pub h_init: GpuBuf,     // [n_heads * d_state]
}

/// Full model weights.
pub struct ModelWeights {
    pub embedding: GpuBuf,        // [vocab, d_model]
    pub final_norm_gamma: GpuBuf, // [d_model]
    pub layers: Vec<MambaLayerWeights>,
}

impl ModelWeights {
    /// Load from a safetensors file, validating every tensor's shape against
    /// `config` so a mismatched checkpoint fails loudly at load time instead of
    /// producing silently-wrong logits.
    pub fn load(path: &str, config: &ModelConfig) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading safetensors file: {path}"))?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parsing safetensors file: {path}"))?;

        let d_model = config.d_model;
        let d_inner = config.d_inner();
        let bc_size = config.bc_size();
        let n_heads = config.n_heads;
        let d_state = config.d_state;
        let in_proj_out = config.in_proj_out();
        let vocab = config.vocab_size;

        let embedding = load_vector(&st, "item.embedding.weight", vocab * d_model)?;
        let final_norm_gamma = load_vector(&st, "item.final_norm.gamma", d_model)?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let p = format!("item.blocks.{i}");
            layers.push(MambaLayerWeights {
                in_proj: load_weight_matrix(
                    &st,
                    &format!("{p}.in_proj.weight"),
                    d_model,
                    in_proj_out,
                )?,
                out_proj: load_weight_matrix(
                    &st,
                    &format!("{p}.out_proj.weight"),
                    d_inner,
                    d_model,
                )?,
                norm_gamma: load_vector(&st, &format!("{p}.norm.gamma"), d_model)?,
                b_gamma: load_vector(&st, &format!("{p}.b_norm.gamma"), bc_size)?,
                c_gamma: load_vector(&st, &format!("{p}.c_norm.gamma"), bc_size)?,
                b_bias: load_vector(&st, &format!("{p}.b_bias"), bc_size)?,
                c_bias: load_vector(&st, &format!("{p}.c_bias"), bc_size)?,
                d_skip: load_vector(&st, &format!("{p}.ssm.d"), n_heads)?,
                dt_bias: load_vector(&st, &format!("{p}.ssm.dt_bias"), n_heads)?,
                h_init: load_vector(&st, &format!("{p}.ssm.h_init"), n_heads * d_state)?,
            });
        }

        Ok(Self { embedding, final_norm_gamma, layers })
    }

    /// Random weights for `lumi-infer smoke` — no checkpoint file required.
    /// Values are small and unstructured; this exercises the kernel pipeline
    /// end to end, not model quality.
    pub fn random(config: &ModelConfig, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let std_dev = 0.02f32;
        fn randn(rng: &mut StdRng, n: usize, scale: f32) -> GpuBuf {
            let data: Vec<f32> = (0..n)
                .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale)
                .collect();
            GpuBuf::from_host(&data)
        }

        let d_model = config.d_model;
        let d_inner = config.d_inner();
        let bc_size = config.bc_size();
        let n_heads = config.n_heads;
        let d_state = config.d_state;
        let in_proj_out = config.in_proj_out();
        let vocab = config.vocab_size;

        let embedding = randn(&mut rng, vocab * d_model, std_dev);
        let final_norm_gamma = GpuBuf::from_host(&vec![1.0f32; d_model]);

        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(MambaLayerWeights {
                in_proj: randn(&mut rng, d_model * in_proj_out, std_dev),
                out_proj: randn(&mut rng, d_inner * d_model, std_dev),
                norm_gamma: GpuBuf::from_host(&vec![1.0f32; d_model]),
                b_gamma: GpuBuf::from_host(&vec![1.0f32; bc_size]),
                c_gamma: GpuBuf::from_host(&vec![1.0f32; bc_size]),
                b_bias: GpuBuf::from_host(&vec![0.0f32; bc_size]),
                c_bias: GpuBuf::from_host(&vec![0.0f32; bc_size]),
                d_skip: GpuBuf::from_host(&vec![1.0f32; n_heads]),
                // dt_bias in a range that keeps softplus(dt+dt_bias) well away
                // from 0 and from blowing up — mirrors native_trainer's random init.
                dt_bias: {
                    let data: Vec<f32> = (0..n_heads)
                        .map(|_| rng.gen::<f32>() * 2.0 - 4.0)
                        .collect();
                    GpuBuf::from_host(&data)
                },
                h_init: GpuBuf::from_host(&vec![0.01f32; n_heads * d_state]),
            });
        }

        Self { embedding, final_norm_gamma, layers }
    }
}

/// Fetch a tensor's raw f32 data and shape, with a dtype check.
fn get_f32_tensor(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let view = st
        .tensor(name)
        .map_err(|e| anyhow::anyhow!("missing tensor '{name}': {e}"))?;
    if view.dtype() != Dtype::F32 {
        bail!("tensor '{name}' has dtype {:?}, expected F32", view.dtype());
    }
    let shape = view.shape().to_vec();
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((data, shape))
}

/// Load a 1D (or flattenable) tensor of exactly `expected_len` elements.
fn load_vector(st: &SafeTensors, name: &str, expected_len: usize) -> Result<GpuBuf> {
    let (data, shape) = get_f32_tensor(st, name)?;
    if data.len() != expected_len {
        bail!(
            "tensor '{name}' has {} elements (shape {:?}), expected {}",
            data.len(),
            shape,
            expected_len
        );
    }
    Ok(GpuBuf::from_host(&data))
}

/// Load a 2D weight matrix stored in Candle/export convention
/// `[out_features, in_features]` and transpose it on the host to the native
/// `[in_features, out_features]` layout `matmul_f32` expects (`C = A @ B`
/// with `B: [k, n] = [in_features, out_features]`).
fn load_weight_matrix(
    st: &SafeTensors,
    name: &str,
    in_features: usize,
    out_features: usize,
) -> Result<GpuBuf> {
    let (data, shape) = get_f32_tensor(st, name)?;
    if shape != [out_features, in_features] {
        bail!(
            "tensor '{name}' has shape {:?}, expected [{out_features}, {in_features}] \
             (out_features, in_features)",
            shape
        );
    }
    // data is row-major [out_features, in_features]; transpose into
    // row-major [in_features, out_features].
    let mut transposed = vec![0.0f32; data.len()];
    for r in 0..out_features {
        for c in 0..in_features {
            transposed[c * out_features + r] = data[r * in_features + c];
        }
    }
    Ok(GpuBuf::from_host(&transposed))
}
