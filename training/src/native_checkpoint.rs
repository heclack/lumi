/// Native checkpoint save/load — GPU buffers to/from raw binary files.
/// Export to safetensors for inference via scripts/export_native_safetensors.py.

#[cfg(feature = "cuda")]
use crate::gpu_memory::{TrainingBuffers, GpuBuf};

/// Save model weights from GPU to checkpoint directory.
/// `meta` is the complete metadata JSON — caller provides all fields (step, loss, epoch, etc.).
#[cfg(feature = "cuda")]
pub fn save_native_checkpoint(
    buf: &TrainingBuffers,
    dir: &str,
    meta: serde_json::Value,
    config: &crate::config::ModelConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    fs::create_dir_all(dir)?;

    let kv_dim = config.attn_kv_dim();
    let mlp_dim = config.attn_mlp_dim();
    let mut n_tensors = 0usize;

    // Helper: write one tensor as raw binary + shape file
    let write_tensor = |name: &str, data: &[f32], shape: &[usize], dir: &str| -> Result<(), Box<dyn std::error::Error>> {
        let safe_name = name.replace('.', "_").replace('/', "_");
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        fs::write(format!("{}/{}.bin", dir, safe_name), bytes)?;
        fs::write(format!("{}/{}.shape", dir, safe_name), format!("{:?}", shape))?;
        Ok(())
    };

    // Embedding + final norm
    write_tensor("item.embedding.weight", &buf.embedding.to_host(), &[buf.vocab, buf.d_model], dir)?;
    write_tensor("item.final_norm.gamma", &buf.final_norm_gamma.to_host(), &[buf.d_model], dir)?;
    n_tensors += 2;

    // Per-layer weights (dispatch by layer type for correct block indexing)
    for (block_idx, layer_type) in buf.layer_types.iter().enumerate() {
        let prefix = format!("item.blocks.{}", block_idx);

        match layer_type {
            crate::gpu_memory::LayerType::Mamba(i) => {
                let w = &buf.layers[*i];
                let bc = buf.n_groups * buf.d_state;
                write_tensor(&format!("{}.in_proj.weight", prefix), &w.in_proj.to_host(), &[buf.d_model, buf.in_proj_out], dir)?;
                write_tensor(&format!("{}.out_proj.weight", prefix), &w.out_proj.to_host(), &[buf.d_inner, buf.d_model], dir)?;
                write_tensor(&format!("{}.norm.gamma", prefix), &w.norm_gamma.to_host(), &[buf.d_model], dir)?;
                write_tensor(&format!("{}.b_norm.gamma", prefix), &w.b_gamma.to_host(), &[bc], dir)?;
                write_tensor(&format!("{}.c_norm.gamma", prefix), &w.c_gamma.to_host(), &[bc], dir)?;
                write_tensor(&format!("{}.ssm.d", prefix), &w.d_skip.to_host(), &[buf.n_heads], dir)?;
                write_tensor(&format!("{}.ssm.dt_bias", prefix), &w.dt_bias.to_host(), &[buf.n_heads], dir)?;
                write_tensor(&format!("{}.b_bias", prefix), &w.b_bias.to_host(), &[bc], dir)?;
                write_tensor(&format!("{}.c_bias", prefix), &w.c_bias.to_host(), &[bc], dir)?;
                write_tensor(&format!("{}.ssm.h_init", prefix), &w.h_init.to_host(), &[buf.n_heads * buf.d_state], dir)?;
                n_tensors += 10;
            }
            crate::gpu_memory::LayerType::Attention(i) => {
                let w = &buf.attn_layers[*i];
                write_tensor(&format!("{}.attn_norm.gamma", prefix), &w.attn_norm_gamma.to_host(), &[buf.d_model], dir)?;
                write_tensor(&format!("{}.q_proj.weight", prefix), &w.q_proj.to_host(), &[buf.d_model, buf.d_model], dir)?;
                write_tensor(&format!("{}.k_proj.weight", prefix), &w.k_proj.to_host(), &[buf.d_model, kv_dim], dir)?;
                write_tensor(&format!("{}.v_proj.weight", prefix), &w.v_proj.to_host(), &[buf.d_model, kv_dim], dir)?;
                write_tensor(&format!("{}.attn_out_proj.weight", prefix), &w.attn_out_proj.to_host(), &[buf.d_model, buf.d_model], dir)?;
                write_tensor(&format!("{}.mlp_norm.gamma", prefix), &w.mlp_norm_gamma.to_host(), &[buf.d_model], dir)?;
                write_tensor(&format!("{}.mlp_gate.weight", prefix), &w.mlp_gate.to_host(), &[buf.d_model, mlp_dim], dir)?;
                write_tensor(&format!("{}.mlp_up.weight", prefix), &w.mlp_up.to_host(), &[buf.d_model, mlp_dim], dir)?;
                write_tensor(&format!("{}.mlp_down.weight", prefix), &w.mlp_down.to_host(), &[mlp_dim, buf.d_model], dir)?;
                n_tensors += 9;
            }
        }
    }

    // Write metadata JSON (single write — caller provides complete metadata)
    let meta_path = format!("{}/meta.json", dir);
    fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

    let step = meta["step"].as_u64().unwrap_or(0);
    let loss = meta["loss"].as_f64().unwrap_or(0.0);
    eprintln!("Native checkpoint saved: {} (step={}, loss={:.4}, {} tensors)",
        dir, step, loss, n_tensors);
    Ok(())
}

/// Load model weights from checkpoint into GPU buffers. Returns step number.
#[cfg(feature = "cuda")]
pub fn load_native_checkpoint(
    buf: &mut TrainingBuffers,
    dir: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    // Parse step from meta.json
    let meta_path = format!("{}/meta.json", dir);
    let step = if let Ok(meta_str) = std::fs::read_to_string(&meta_path) {
        let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
        meta["step"].as_u64().unwrap_or(0) as usize
    } else {
        0
    };

    // Load embedding
    load_tensor_bin(&format!("{}/item_embedding_weight.bin", dir), &mut buf.embedding)?;
    load_tensor_bin(&format!("{}/item_final_norm_gamma.bin", dir), &mut buf.final_norm_gamma)?;

    for (block_idx, layer_type) in buf.layer_types.iter().enumerate() {
        let prefix = format!("item_blocks_{}", block_idx);
        match layer_type {
            crate::gpu_memory::LayerType::Mamba(i) => {
                let layer = &mut buf.layers[*i];
                load_tensor_bin(&format!("{}/{}_in_proj_weight.bin", dir, prefix), &mut layer.in_proj)?;
                load_tensor_bin(&format!("{}/{}_out_proj_weight.bin", dir, prefix), &mut layer.out_proj)?;
                load_tensor_bin(&format!("{}/{}_norm_gamma.bin", dir, prefix), &mut layer.norm_gamma)?;
                load_tensor_bin(&format!("{}/{}_b_norm_gamma.bin", dir, prefix), &mut layer.b_gamma)?;
                load_tensor_bin(&format!("{}/{}_c_norm_gamma.bin", dir, prefix), &mut layer.c_gamma)?;
                load_tensor_bin(&format!("{}/{}_ssm_d.bin", dir, prefix), &mut layer.d_skip)?;
                load_tensor_bin(&format!("{}/{}_ssm_dt_bias.bin", dir, prefix), &mut layer.dt_bias)?;
                load_tensor_bin(&format!("{}/{}_b_bias.bin", dir, prefix), &mut layer.b_bias)?;
                load_tensor_bin(&format!("{}/{}_c_bias.bin", dir, prefix), &mut layer.c_bias)?;
                load_tensor_bin(&format!("{}/{}_ssm_h_init.bin", dir, prefix), &mut layer.h_init)?;
            }
            crate::gpu_memory::LayerType::Attention(i) => {
                let aw = &mut buf.attn_layers[*i];
                load_tensor_bin(&format!("{}/{}_attn_norm_gamma.bin", dir, prefix), &mut aw.attn_norm_gamma)?;
                load_tensor_bin(&format!("{}/{}_q_proj_weight.bin", dir, prefix), &mut aw.q_proj)?;
                load_tensor_bin(&format!("{}/{}_k_proj_weight.bin", dir, prefix), &mut aw.k_proj)?;
                load_tensor_bin(&format!("{}/{}_v_proj_weight.bin", dir, prefix), &mut aw.v_proj)?;
                load_tensor_bin(&format!("{}/{}_attn_out_proj_weight.bin", dir, prefix), &mut aw.attn_out_proj)?;
                load_tensor_bin(&format!("{}/{}_mlp_norm_gamma.bin", dir, prefix), &mut aw.mlp_norm_gamma)?;
                load_tensor_bin(&format!("{}/{}_mlp_gate_weight.bin", dir, prefix), &mut aw.mlp_gate)?;
                load_tensor_bin(&format!("{}/{}_mlp_up_weight.bin", dir, prefix), &mut aw.mlp_up)?;
                load_tensor_bin(&format!("{}/{}_mlp_down_weight.bin", dir, prefix), &mut aw.mlp_down)?;
            }
        }
    }

    eprintln!("Native checkpoint loaded from: {} (step {})", dir, step);
    Ok(step)
}

#[cfg(feature = "cuda")]
fn load_tensor_bin(path: &str, buf: &mut GpuBuf) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let data: Vec<f32> = bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if data.len() == buf.len {
        buf.copy_from_host(&data);
    } else if data.len() < buf.len {
        // Checkpoint tensor is smaller (e.g., in_proj grew from DD-RoPE).
        // Load what we have, leave remaining elements at their initialized values.
        let mut full = vec![0.0f32; buf.len];
        full[..data.len()].copy_from_slice(&data);
        // Copy current buf values for the extra portion (preserves random init)
        let current = buf.to_host();
        full[data.len()..].copy_from_slice(&current[data.len()..]);
        *buf = GpuBuf::from_host(&full);
        eprintln!("  Partial load: {} has {} values, buffer expects {} — extra columns keep init",
            path, data.len(), buf.len);
    } else {
        return Err(format!("Tensor too large: {} has {} values, buffer only {}", path, data.len(), buf.len).into());
    }
    Ok(())
}

/// Write a GPU float buffer directly to a BufWriter as little-endian bytes, no intermediate Vec<u8>.
/// Safety: x86-64 is little-endian, so f32 raw memory layout == LE bytes.
#[cfg(feature = "cuda")]
fn write_gpu_floats(writer: &mut std::io::BufWriter<std::fs::File>, floats: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let bytes = unsafe {
        std::slice::from_raw_parts(floats.as_ptr() as *const u8, floats.len() * 4)
    };
    writer.write_all(bytes)
}

/// Save optimizer state (Adam m/v) for resume without momentum loss.
/// Writes a single concatenated binary: all m states then all v states, in deterministic order.
/// Streams directly via BufWriter to avoid holding a full second copy of the state in CPU RAM.
#[cfg(feature = "cuda")]
pub fn save_optimizer_state(
    buf: &TrainingBuffers,
    dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let path = format!("{}/optimizer.bin", dir);
    let file = std::fs::File::create(&path)?;
    let mut writer = std::io::BufWriter::with_capacity(8 * 1024 * 1024, file);
    let mut total_floats: usize = 0;

    // Mamba large params (in_proj + out_proj) — m then v per layer
    for adam in &buf.adam_m {
        let v = adam.m.to_host(); total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
        let v = adam.v.to_host(); total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
    }

    // Mamba small params — contiguous m_buf and v_buf per layer
    for sa in &buf.small_adam {
        let v = sa.m_buf.to_host(); total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
        let v = sa.v_buf.to_host(); total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
    }

    // Attention Adam state
    for aa in &buf.attn_adam {
        let v = aa.m.to_host();          total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
        let v = aa.v.to_host();          total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
        let v = aa.norm_m.to_host();     total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
        let v = aa.norm_v.to_host();     total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
        let v = aa.mlp_norm_m.to_host(); total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
        let v = aa.mlp_norm_v.to_host(); total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
    }

    // Global Adam state
    let v = buf.final_norm_adam_m.to_host(); total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
    let v = buf.final_norm_adam_v.to_host(); total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
    let v = buf.embedding_adam_m.to_host();  total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;
    let v = buf.embedding_adam_v.to_host();  total_floats += v.len(); write_gpu_floats(&mut writer, &v)?;

    writer.flush()?;
    eprintln!("  Optimizer state saved: {} ({:.1}MB)", path, total_floats as f64 * 4.0 / 1e6);
    Ok(())
}

/// Load optimizer state from checkpoint. Skips if file doesn't exist (fresh training start).
#[cfg(feature = "cuda")]
pub fn load_optimizer_state(
    buf: &mut TrainingBuffers,
    dir: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let path = format!("{}/optimizer.bin", dir);
    if !std::path::Path::new(&path).exists() {
        eprintln!("  No optimizer state found — Adam restarts from zeros");
        return Ok(false);
    }

    let bytes = std::fs::read(&path)?;
    let data: Vec<f32> = bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut offset = 0usize;

    // Helper: copy data into existing GpuBuf (preserves sub-pointers in SmallParamAdam)
    let mut load_into = |buf: &GpuBuf| {
        let n = buf.len;
        if offset + n <= data.len() {
            buf.copy_from_host(&data[offset..offset + n]);
            offset += n;
        }
    };

    // Mamba large params
    for adam in &buf.adam_m {
        load_into(&adam.m);
        load_into(&adam.v);
    }

    // Mamba small params — copy into existing contiguous allocation (raw sub-pointers stay valid)
    for sa in &buf.small_adam {
        load_into(&sa.m_buf);
        load_into(&sa.v_buf);
    }

    // Attention Adam state
    for aa in &buf.attn_adam {
        load_into(&aa.m);
        load_into(&aa.v);
        load_into(&aa.norm_m); load_into(&aa.norm_v);
        load_into(&aa.mlp_norm_m); load_into(&aa.mlp_norm_v);
    }

    // Global Adam state
    load_into(&buf.final_norm_adam_m);
    load_into(&buf.final_norm_adam_v);
    load_into(&buf.embedding_adam_m);
    load_into(&buf.embedding_adam_v);

    eprintln!("  Optimizer state loaded: {} ({:.1}MB, {} values)", path, data.len() as f64 * 4.0 / 1e6, offset);
    Ok(true)
}

// Stubs for non-CUDA builds (never called, but must exist for module to compile)
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub fn save_native_checkpoint(
    _buf: &(), _dir: &str, _meta: serde_json::Value, _config: &crate::config::ModelConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    panic!("Native checkpoints require --features cuda")
}
