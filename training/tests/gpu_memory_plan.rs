//! Measures the real device footprint of TrainingBuffers and reports what fits.
//!
//! Run with:  cargo test --features hip --release --test gpu_memory_plan -- --nocapture
//!
//! ── The model ───────────────────────────────────────────────────────────────
//! Training memory splits into two parts that scale differently:
//!
//!   Total(batch, seq)  =  A  +  B * (batch * seq)
//!
//!   A = weights + gradients + AdamW moments + vocab-sized buffers.
//!       Independent of batch/seq. In fp32 AdamW this is ~16 bytes per
//!       parameter: 4 (weight) + 4 (grad) + 4 (m) + 4 (v).
//!
//!   B = per-token activation cost, summed over every buffer sized bs * D for
//!       some feature width D. This is what batch_size and seq_len actually buy.
//!
//! Because the relationship is affine, two measurements pin down both constants
//! exactly -- no need to hand-audit sixty allocation sites (and no risk of the
//! audit drifting from the code, which is how the theta bug in the scan kernel
//! survived). A third point is measured and used to verify linearity.

#![cfg(feature = "gpu")]

use lumi::config::ModelConfig;
use lumi::gpu_memory::{format_bytes, gpu_bytes_live, TrainingBuffers};

/// Physical VRAM on the target card (Radeon AI PRO R9700), minus a working
/// margin for the driver, fragmentation, and kernel scratch. The scan kernels
/// allocate ~2 KB/thread of private scratch, which lives outside these buffers.
const VRAM_TOTAL: usize = 34_208_743_424; // 31.86 GiB, from rocm-smi
const RESERVE: usize = 2 << 30; // 2 GiB headroom
const BUDGET: usize = VRAM_TOTAL - RESERVE;

/// Allocate, measure, free. Returns bytes held by TrainingBuffers alone.
fn measure_with(
    config: &ModelConfig,
    batch: usize,
    seq: usize,
    mixed_precision: bool,
    bf16_activations: bool,
) -> usize {
    let before = gpu_bytes_live();
    let buffers = TrainingBuffers::allocate(config, batch, seq, mixed_precision, bf16_activations);
    let held = gpu_bytes_live() - before;
    drop(buffers);
    assert_eq!(
        gpu_bytes_live(),
        before,
        "TrainingBuffers leaked device memory on drop"
    );
    held
}

fn measure(config: &ModelConfig, batch: usize, seq: usize) -> usize {
    measure_with(config, batch, seq, false, false)
}

/// Fit Total = A + B*tokens from two probes; returns (fixed, per_token).
fn fit(config: &ModelConfig, mixed_precision: bool, bf16_activations: bool) -> (f64, f64) {
    let m0 = measure_with(config, 1, 128, mixed_precision, bf16_activations) as f64;
    let m1 = measure_with(config, 1, 512, mixed_precision, bf16_activations) as f64;
    let per_token = (m1 - m0) / (512.0 - 128.0);
    (m0 - per_token * 128.0, per_token)
}

#[test]
fn report_memory_footprint_and_fit() {
    let config = ModelConfig::default();
    let params = config.param_count();

    eprintln!("\n  ── model ──────────────────────────────────────────────");
    eprintln!("  d_model={}  n_layers={}  d_state={}  n_heads={}", config.d_model, config.n_layers, config.d_state, config.n_heads);
    eprintln!("  d_inner={}  vocab={}  attn_layers={}", config.d_inner(), config.vocab_size, config.n_attn_layers());
    eprintln!("  parameters: {:.1}M", params as f64 / 1e6);
    eprintln!("  fp32 AdamW lower bound (16 B/param): {}", format_bytes(params * 16));

    // Three (batch, seq) points with distinct token counts. Small enough that
    // all three fit comfortably regardless of the fixed cost.
    let probes = [(1usize, 128usize), (1, 256), (2, 256)];
    let mut measured = Vec::new();

    eprintln!("\n  ── measured ───────────────────────────────────────────");
    eprintln!("  batch  seq   tokens        footprint");
    for &(b, s) in &probes {
        let bytes = measure(&config, b, s);
        eprintln!("  {:>5} {:>5} {:>7}   {:>14}", b, s, b * s, format_bytes(bytes));
        measured.push((b * s, bytes));
    }

    // Solve Total = A + B * tokens from the first two points.
    let (t0, m0) = measured[0];
    let (t1, m1) = measured[1];
    assert!(t1 > t0, "probe token counts must differ");
    let per_token = (m1 - m0) as f64 / (t1 - t0) as f64;
    let fixed = m0 as f64 - per_token * t0 as f64;

    // Verify linearity against the held-out third point.
    let (t2, m2) = measured[2];
    let predicted = fixed + per_token * t2 as f64;
    let rel_err = ((predicted - m2 as f64) / m2 as f64).abs();

    eprintln!("\n  ── fitted model ───────────────────────────────────────");
    eprintln!("  fixed cost A          : {}", format_bytes(fixed as usize));
    eprintln!("  per-token cost B      : {:.1} KiB/token", per_token / 1024.0);
    eprintln!("  check @ {} tokens   : predicted {}, actual {} ({:.2}% off)",
        t2, format_bytes(predicted as usize), format_bytes(m2), rel_err * 100.0);

    assert!(
        rel_err < 0.02,
        "footprint is not affine in token count ({:.1}% error) -- the A + B*tokens \
         model above is wrong and the projections below cannot be trusted",
        rel_err * 100.0
    );

    // What fits in the budget?
    let token_budget = ((BUDGET as f64 - fixed) / per_token) as usize;
    eprintln!("\n  ── capacity on {} VRAM ─────────────────────", format_bytes(VRAM_TOTAL));
    eprintln!("  usable budget         : {} (after {} reserve)", format_bytes(BUDGET), format_bytes(RESERVE));
    eprintln!("  fixed cost            : {} ({:.0}% of budget)", format_bytes(fixed as usize), 100.0 * fixed / BUDGET as f64);
    eprintln!("  tokens affordable     : {}", token_budget);

    eprintln!("\n  seq_len   max batch   tokens/step   footprint");
    for seq in [256usize, 512, 1024, 2048] {
        let max_batch = token_budget / seq;
        if max_batch == 0 {
            eprintln!("  {:>7}   {:>9}   {:>11}   does not fit", seq, "-", "-");
            continue;
        }
        let tokens = max_batch * seq;
        let bytes = fixed + per_token * tokens as f64;
        eprintln!("  {:>7}   {:>9}   {:>11}   {:>10}", seq, max_batch, tokens, format_bytes(bytes as usize));
    }

    // Where does the per-token cost go? Vary depth: the per-layer saved
    // activations are the only per-token term that scales with n_layers, so
    // differencing two depths splits B into per-layer and depth-independent
    // parts without auditing every allocation.
    let mut shallow = config.clone();
    shallow.n_layers = 8;
    let (_, per_token_shallow) = fit(&shallow, false, false);
    let per_layer =
        (per_token - per_token_shallow) / (config.n_layers - shallow.n_layers) as f64;
    let depth_independent = per_token - per_layer * config.n_layers as f64;

    eprintln!("\n  ── per-token cost breakdown ───────────────────────────");
    eprintln!("  per layer             : {:.1} KiB/token", per_layer / 1024.0);
    eprintln!("  x {} layers            : {:.1} KiB/token ({:.0}%)",
        config.n_layers,
        per_layer * config.n_layers as f64 / 1024.0,
        100.0 * per_layer * config.n_layers as f64 / per_token);
    eprintln!("  depth-independent     : {:.1} KiB/token ({:.0}%)",
        depth_independent / 1024.0,
        100.0 * depth_independent / per_token);

    // What the precision flags actually buy.
    eprintln!("\n  ── effect of precision flags ──────────────────────────");
    eprintln!("  mixed  bf16_act   fixed        per-token      tokens affordable");
    for &(mp, ba) in &[(false, false), (false, true), (true, false), (true, true)] {
        let (f, pt) = fit(&config, mp, ba);
        let budget_tokens = ((BUDGET as f64 - f) / pt) as usize;
        eprintln!(
            "  {:<6} {:<9} {:>10}   {:>8.1} KiB   {:>10}",
            mp, ba, format_bytes(f as usize), pt / 1024.0, budget_tokens
        );
    }

    // Concrete settings that preserve the default effective batch. batch_size is
    // the MICRO-batch (native_trainer.rs runs `for _micro in 0..grad_accum`, each
    // a full batch_size pass), and the LR is already divided by grad_accum -- so
    // trading micro-batch for accumulation leaves the optimizer math unchanged.
    let effective_seqs = 32 * 4; // default batch_size * gradient_accumulation
    eprintln!("\n  ── settings preserving effective batch of {} sequences ──", effective_seqs);
    eprintln!("  seq_len   batch_size   grad_accum   footprint     fits?");
    for seq in [512usize, 1024, 2048] {
        let max_batch = token_budget / seq;
        // Largest power-of-two micro-batch that fits and divides the effective batch.
        let micro = (1..=max_batch.min(effective_seqs))
            .rev()
            .find(|m| effective_seqs % m == 0)
            .unwrap_or(0);
        if micro == 0 {
            eprintln!("  {:>7}   {:>10}   {:>10}   {:>11}   NO", seq, "-", "-", "-");
            continue;
        }
        let bytes = fixed + per_token * (micro * seq) as f64;
        eprintln!(
            "  {:>7}   {:>10}   {:>10}   {:>11}   {}",
            seq,
            micro,
            effective_seqs / micro,
            format_bytes(bytes as usize),
            if bytes <= BUDGET as f64 { "yes" } else { "NO" }
        );
    }
    eprintln!();
}

/// The projections above are extrapolations. This actually allocates the
/// recommended configuration on the real card, so the recommendation is proven
/// rather than predicted.
#[test]
fn recommended_config_actually_fits() {
    let config = ModelConfig::default();

    // Default is batch_size 32 x seq 2048 = 65536 tokens. Recommended micro-batch
    // is 2 x 2048 = 4096 tokens, with gradient_accumulation raised 4 -> 64 to hold
    // the effective batch at 128 sequences.
    let (batch, seq) = (2usize, 2048usize);

    let fp32 = measure_with(&config, batch, seq, false, false);
    eprintln!(
        "\n  recommended batch={} seq={} (fp32 activations): {}",
        batch, seq, format_bytes(fp32)
    );
    assert!(
        fp32 <= BUDGET,
        "recommended config needs {} but budget is {}",
        format_bytes(fp32),
        format_bytes(BUDGET)
    );

    // With bf16 activations the same budget buys a larger micro-batch.
    let bf16 = measure_with(&config, batch * 2, seq, false, true);
    eprintln!(
        "  batch={} seq={} (bf16 activations):        {}\n",
        batch * 2, seq, format_bytes(bf16)
    );
    assert!(
        bf16 <= BUDGET,
        "bf16 config needs {} but budget is {}",
        format_bytes(bf16),
        format_bytes(BUDGET)
    );
}
