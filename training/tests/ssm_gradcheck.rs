//! Numerical gradient check for the SSM selective scan backward kernel.
//!
//! Requires a GPU backend:  cargo test --features hip --test ssm_gradcheck
//!
//! ── What this verifies ──────────────────────────────────────────────────────
//! Pair the scan output with a fixed random covector w to get a scalar loss:
//!
//!     L = <y, w> = sum_i y_i w_i      =>      dL/dy = w   (exactly)
//!
//! Feeding that w in as `dy`, the backward kernel claims to produce dL/dtheta
//! for every input tensor theta. If J = dy/dtheta is the forward Jacobian, then
//! by the chain rule dL/dtheta = J^T w -- i.e. backward is the adjoint
//! (transpose) of the forward's linearization.
//!
//! Central differences give an independent O(eps^2) estimate of the same thing:
//!
//!     dL/dtheta_j ~= [ L(theta + eps*e_j) - L(theta - eps*e_j) ] / (2 eps)
//!
//! Agreement between the two is strong evidence the backward pass is correct.
//! A random w (rather than all-ones) is deliberate: with all-ones, sign errors
//! in J^T can cancel inside the sum and go undetected.
//!
//! This is the first execution of ssm_scan_bwd_gpu_v2 on the AMD/HIP port, so
//! it also exercises the hand-ported warp-shuffle reductions and the
//! __cosf/__sinf DD-RoPE path that replaced NVIDIA PTX intrinsics.

#![cfg(feature = "gpu")]

use lumi::gpu_memory::GpuBuf;
use lumi::native_ops;
use std::ptr;

// D_STATE MUST BE 64. The backward kernel's DD-RoPE theta reduction runs under
// `if (tid < half_d_state)` and then reduces with a full 32-lane warp mask
// (ssm_scan.cu, "Warp-shuffle reduce across tid 0..half_d_state-1"). That is
// only correct when half_d_state == 32 exactly, i.e. d_state == 64 -- which is
// the configured default. With a smaller d_state the mask names more lanes than
// are active: NVIDIA silently reduces garbage, and HIP traps outright
// (HSA_STATUS_ERROR_EXCEPTION), because __hip_check_mask asserts
// mask == __ballot(true). Do not shrink D_STATE to make this test faster.
//
// Otherwise kept small: >1 head, >1 chunk, and head_dim chosen so that
// state_size (= head_dim * d_state = 512) exceeds the 256-thread block, giving
// max_elems == 2 and exercising the multi-element-per-thread path.
const BATCH: usize = 1;
const SEQ: usize = 8;
const N_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const D_STATE: usize = 64;
const N_GROUPS: usize = 1;
const CHUNK: usize = 4; // backward chunk size; kernel accepts 4, 8, 16, 32

/// Cap on finite-differenced elements per tensor. Each probe costs two forward
/// launches, so large tensors are sampled rather than swept exhaustively.
const MAX_PROBES: usize = 32;

const N_X: usize = BATCH * SEQ * N_HEADS * HEAD_DIM;
const N_DT: usize = BATCH * SEQ * N_HEADS;
const N_BC: usize = BATCH * SEQ * N_GROUPS * D_STATE;
const N_HEAD_P: usize = N_HEADS;
const N_H_INIT: usize = N_HEADS * D_STATE;
const N_THETA: usize = BATCH * SEQ * N_HEADS * (D_STATE / 2);
const N_Y: usize = N_X;

/// Deterministic LCG so failures are reproducible run to run.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    /// Uniform in [-1, 1).
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

/// The scan's input tensors, in host memory.
#[derive(Clone)]
struct Inputs {
    x: Vec<f32>,
    dt: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d_skip: Vec<f32>,
    dt_bias: Vec<f32>,
    h_init: Vec<f32>,
    lambda: Vec<f32>,
    theta: Vec<f32>,
    a_vals: Vec<f32>,
}

impl Inputs {
    /// Values chosen to sit in the same regime the trainer produces, so the
    /// gradient check exercises realistic numerics rather than a degenerate
    /// corner (e.g. a_bar ~ 0 would zero out the recurrence entirely).
    fn sample(rng: &mut Rng) -> Self {
        Inputs {
            x: rng.vec(N_X, 1.0),
            dt: rng.vec(N_DT, 0.5),
            b: rng.vec(N_BC, 1.0),
            c: rng.vec(N_BC, 1.0),
            d_skip: rng.vec(N_HEAD_P, 1.0),
            dt_bias: rng.vec(N_HEAD_P, 0.5),
            h_init: rng.vec(N_H_INIT, 0.5),
            // lambda is a sigmoid output in the trainer => strictly in (0, 1).
            lambda: (0..N_DT).map(|_| 0.5 + 0.25 * rng.next()).collect(),
            // theta is raw; forward applies tanh(theta) * pi internally.
            theta: rng.vec(N_THETA, 1.0),
            // A_vals come from neg_softplus_clamp => strictly negative, so
            // a_bar = exp(A * dt_pos) is a decay factor in (0, 1).
            a_vals: (0..N_DT).map(|_| -0.5 - 0.4 * (rng.next().abs())).collect(),
        }
    }

    /// Mutable view of one named tensor, for perturbation.
    fn tensor_mut(&mut self, which: Tensor) -> &mut Vec<f32> {
        match which {
            Tensor::X => &mut self.x,
            Tensor::Dt => &mut self.dt,
            Tensor::B => &mut self.b,
            Tensor::C => &mut self.c,
            Tensor::DSkip => &mut self.d_skip,
            Tensor::DtBias => &mut self.dt_bias,
            Tensor::HInit => &mut self.h_init,
            Tensor::Lambda => &mut self.lambda,
            Tensor::Theta => &mut self.theta,
            Tensor::AVals => &mut self.a_vals,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Tensor {
    X,
    Dt,
    B,
    C,
    DSkip,
    DtBias,
    HInit,
    Lambda,
    Theta,
    AVals,
}

impl Tensor {
    fn name(self) -> &'static str {
        match self {
            Tensor::X => "x",
            Tensor::Dt => "dt",
            Tensor::B => "B",
            Tensor::C => "C",
            Tensor::DSkip => "D (skip)",
            Tensor::DtBias => "dt_bias",
            Tensor::HInit => "h_init",
            Tensor::Lambda => "lambda",
            Tensor::Theta => "theta",
            Tensor::AVals => "A_vals",
        }
    }
}

const ALL_TENSORS: [Tensor; 10] = [
    Tensor::X,
    Tensor::Dt,
    Tensor::B,
    Tensor::C,
    Tensor::DSkip,
    Tensor::DtBias,
    Tensor::HInit,
    Tensor::Lambda,
    Tensor::Theta,
    Tensor::AVals,
];

/// Run the forward scan and return L = <y, w>.
fn forward_loss(inp: &Inputs, w: &[f32]) -> f32 {
    let x = GpuBuf::from_host(&inp.x);
    let dt = GpuBuf::from_host(&inp.dt);
    let b = GpuBuf::from_host(&inp.b);
    let c = GpuBuf::from_host(&inp.c);
    let d_skip = GpuBuf::from_host(&inp.d_skip);
    let dt_bias = GpuBuf::from_host(&inp.dt_bias);
    let h_init = GpuBuf::from_host(&inp.h_init);
    let lambda = GpuBuf::from_host(&inp.lambda);
    let theta = GpuBuf::from_host(&inp.theta);
    let a_vals = GpuBuf::from_host(&inp.a_vals);
    let y = GpuBuf::alloc(N_Y);
    y.zero();

    unsafe {
        native_ops::ssm_scan_fwd_gpu(
            x.ptr,
            dt.ptr,
            b.ptr,
            c.ptr,
            d_skip.ptr,
            dt_bias.ptr,
            h_init.ptr,
            lambda.ptr,
            theta.ptr,
            a_vals.ptr,
            ptr::null(),     // z_in: no gating, keeps the map purely the scan
            y.ptr,
            ptr::null_mut(), // y_gated: unused
            BATCH as i32,
            SEQ as i32,
            N_HEADS as i32,
            HEAD_DIM as i32,
            D_STATE as i32,
            N_GROUPS as i32,
        );
        native_ops::cudaDeviceSynchronize();
    }

    let y_host = y.to_host();
    // Accumulate the inner product in f64: with ~64 terms, f32 summation error
    // would otherwise leak into the finite-difference quotient.
    y_host.iter().zip(w).map(|(&yi, &wi)| yi as f64 * wi as f64).sum::<f64>() as f32
}

/// Analytic gradients from the backward kernel, keyed by tensor.
struct Grads {
    dx: Vec<f32>,
    ddt: Vec<f32>,
    db: Vec<f32>,
    dc: Vec<f32>,
    d_d_skip: Vec<f32>,
    d_dt_bias: Vec<f32>,
    d_h_init: Vec<f32>,
    d_lambda: Vec<f32>,
    d_theta: Vec<f32>,
    d_a_vals: Vec<f32>,
}

impl Grads {
    fn get(&self, which: Tensor) -> &[f32] {
        match which {
            Tensor::X => &self.dx,
            Tensor::Dt => &self.ddt,
            Tensor::B => &self.db,
            Tensor::C => &self.dc,
            Tensor::DSkip => &self.d_d_skip,
            Tensor::DtBias => &self.d_dt_bias,
            Tensor::HInit => &self.d_h_init,
            Tensor::Lambda => &self.d_lambda,
            Tensor::Theta => &self.d_theta,
            Tensor::AVals => &self.d_a_vals,
        }
    }
}

/// Run the backward pass with dy = w, returning all analytic gradients.
fn backward_grads(inp: &Inputs, w: &[f32]) -> Grads {
    let x = GpuBuf::from_host(&inp.x);
    let dt = GpuBuf::from_host(&inp.dt);
    let b = GpuBuf::from_host(&inp.b);
    let c = GpuBuf::from_host(&inp.c);
    let d_skip = GpuBuf::from_host(&inp.d_skip);
    let dt_bias = GpuBuf::from_host(&inp.dt_bias);
    let h_init = GpuBuf::from_host(&inp.h_init);
    let lambda = GpuBuf::from_host(&inp.lambda);
    let theta = GpuBuf::from_host(&inp.theta);
    let a_vals = GpuBuf::from_host(&inp.a_vals);
    let dy = GpuBuf::from_host(w);

    // Workspace sizing mirrors GpuMemory::new in src/gpu_memory.rs.
    let n_threads = 256;
    let state_size = HEAD_DIM * D_STATE;
    let max_elems = (state_size + n_threads - 1) / n_threads;
    let n_chunks = (SEQ + CHUNK - 1) / CHUNK;
    let ckpt_size = BATCH * N_HEADS * n_chunks * n_threads * max_elems;
    let saved_size = BATCH * N_HEADS * CHUNK * n_threads * max_elems;

    let h_ckpt = GpuBuf::alloc(ckpt_size);
    let pbx_ckpt = GpuBuf::alloc(ckpt_size);
    let h_saved = GpuBuf::alloc(saved_size);
    let pbx_saved = GpuBuf::alloc(saved_size);

    let dx = GpuBuf::alloc(N_X);
    let ddt = GpuBuf::alloc(N_DT);
    let db = GpuBuf::alloc(N_BC);
    let dc = GpuBuf::alloc(N_BC);
    let d_lambda = GpuBuf::alloc(N_DT);
    let d_h_init = GpuBuf::alloc(N_H_INIT);
    let d_theta = GpuBuf::alloc(N_THETA);
    let d_a_vals = GpuBuf::alloc(N_DT);
    let d_d_skip = GpuBuf::alloc(N_HEAD_P);
    let d_dt_bias = GpuBuf::alloc(N_HEAD_P);
    let ws_dd = GpuBuf::alloc(BATCH * N_HEADS);
    let ws_dtb = GpuBuf::alloc(BATCH * N_HEADS);

    // dD and d_dt_bias are written with atomicAdd (the trainer accumulates them
    // across layers and micro-batches), so they MUST start at zero here. The
    // rest are zeroed defensively -- the kernel memsets some of them itself.
    for buf in [
        &dx, &ddt, &db, &dc, &d_lambda, &d_h_init, &d_theta, &d_a_vals, &d_d_skip, &d_dt_bias,
        &ws_dd, &ws_dtb, &h_ckpt, &pbx_ckpt, &h_saved, &pbx_saved,
    ] {
        buf.zero();
    }

    unsafe {
        native_ops::ssm_scan_bwd_gpu_v2(
            x.ptr,
            dt.ptr,
            b.ptr,
            c.ptr,
            d_skip.ptr,
            dt_bias.ptr,
            dy.ptr,
            lambda.ptr,
            h_init.ptr,
            h_ckpt.ptr,
            pbx_ckpt.ptr,
            h_saved.ptr,
            pbx_saved.ptr,
            theta.ptr,
            a_vals.ptr,
            dx.ptr,
            ddt.ptr,
            db.ptr,
            dc.ptr,
            d_lambda.ptr,
            d_h_init.ptr,
            d_theta.ptr,
            d_a_vals.ptr,
            d_d_skip.ptr,
            d_dt_bias.ptr,
            ws_dd.ptr,
            ws_dtb.ptr,
            BATCH as i32,
            SEQ as i32,
            N_HEADS as i32,
            HEAD_DIM as i32,
            D_STATE as i32,
            N_GROUPS as i32,
            CHUNK as i32,
        );
        native_ops::cudaDeviceSynchronize();
    }

    Grads {
        dx: dx.to_host(),
        ddt: ddt.to_host(),
        db: db.to_host(),
        dc: dc.to_host(),
        d_d_skip: d_d_skip.to_host(),
        d_dt_bias: d_dt_bias.to_host(),
        d_h_init: d_h_init.to_host(),
        d_lambda: d_lambda.to_host(),
        d_theta: d_theta.to_host(),
        d_a_vals: d_a_vals.to_host(),
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Which elements of a length-`n` tensor to finite-difference. Small tensors are
/// swept exhaustively; larger ones are walked with a stride coprime to `n` from a
/// pseudo-random start, so the probes spread across all index residues rather
/// than clustering on one (e.g. always hitting state index s == 0).
fn probe_indices(n: usize, rng: &mut Rng) -> Vec<usize> {
    if n <= MAX_PROBES {
        return (0..n).collect();
    }
    let mut stride = n / 3 + 1;
    while gcd(stride, n) != 1 {
        stride += 1;
    }
    let start = ((rng.next().abs() * n as f32) as usize) % n;
    (0..MAX_PROBES).map(|i| (start + i * stride) % n).collect()
}

/// Central-difference estimate of dL/dtheta at the given element indices.
fn numerical_grad(inp: &Inputs, w: &[f32], which: Tensor, idxs: &[usize], eps: f32) -> Vec<f32> {
    idxs.iter()
        .map(|&j| {
            let mut plus = inp.clone();
            plus.tensor_mut(which)[j] += eps;
            let mut minus = inp.clone();
            minus.tensor_mut(which)[j] -= eps;

            let l_plus = forward_loss(&plus, w) as f64;
            let l_minus = forward_loss(&minus, w) as f64;
            ((l_plus - l_minus) / (2.0 * eps as f64)) as f32
        })
        .collect()
}

/// Worst relative error over the probed indices.
///
/// The denominator carries an absolute floor tied to the tensor's own gradient
/// scale, because f32 finite differences cannot resolve small entries. The
/// forward loss L is O(10-100) with an f32 ULP around 4e-6, while a single
/// probe moves it by only 2*eps*grad; for a gradient two orders of magnitude
/// below the tensor's largest, that difference is a handful of ULPs and the
/// quotient is mostly quantisation noise. Judging such entries on pure relative
/// error would report failures that shrink when eps grows -- the signature of
/// cancellation noise, not a wrong kernel. Their absolute contribution to any
/// weight update is negligible anyway.
fn max_rel_err(analytic: &[f32], numeric: &[f32], idxs: &[usize]) -> (f32, usize) {
    assert_eq!(numeric.len(), idxs.len(), "probe count mismatch");
    let scale = idxs
        .iter()
        .zip(numeric)
        .map(|(&j, n)| analytic[j].abs().max(n.abs()))
        .fold(0.0f32, f32::max);
    let floor = (0.02 * scale).max(1e-4);

    let mut worst = 0.0f32;
    let mut worst_idx = idxs[0];
    for (k, &j) in idxs.iter().enumerate() {
        let (a, n) = (analytic[j], numeric[k]);
        let denom = a.abs().max(n.abs()).max(floor);
        let rel = (a - n).abs() / denom;
        if rel > worst {
            worst = rel;
            worst_idx = j;
        }
    }
    (worst, worst_idx)
}

/// Core gradient check. `zero_theta` sets the DD-RoPE angles to zero, which
/// makes the B/C rotation the identity and removes theta's contribution to the
/// dt gradient -- isolating the a_bar (decay) path from the DD-RoPE path.
/// Returns the per-tensor max relative errors.
fn run_gradcheck(zero_theta: bool, tol: f32) -> Vec<(Tensor, f32)> {
    let mut rng = Rng::new(0x5eed_1234);
    let mut inp = Inputs::sample(&mut rng);
    if zero_theta {
        inp.theta = vec![0.0; N_THETA];
    }
    let w = rng.vec(N_Y, 1.0); // the random covector defining L = <y, w>

    // Sanity: the forward pass must produce finite output, or the whole
    // comparison below is vacuous.
    let l0 = forward_loss(&inp, &w);
    assert!(l0.is_finite(), "forward loss is not finite: {}", l0);

    let analytic = backward_grads(&inp, &w);

    // eps balances truncation error O(eps^2) against f32 cancellation O(1/eps).
    // 1e-2 was picked empirically: dropping to 2e-3 raises the error on every
    // small-magnitude tensor (C went 4.6e-2 -> 4.3e-1), which is the cancellation
    // side dominating. Raising it further starts to show curvature instead.
    let eps = 1e-2f32;

    let mut errs = Vec::new();
    let mut failures = Vec::new();
    eprintln!(
        "\n  === gradcheck (DD-RoPE {}) ===",
        if zero_theta { "OFF: theta = 0" } else { "ON" }
    );
    eprintln!("  tensor      probes/n   max_rel_err     analytic      numeric   @idx");
    eprintln!("  --------------------------------------------------------------------");
    for t in ALL_TENSORS {
        let ana = analytic.get(t);
        let idxs = probe_indices(ana.len(), &mut rng);
        let num = numerical_grad(&inp, &w, t, &idxs, eps);
        let (err, idx) = max_rel_err(ana, &num, &idxs);
        let at = idxs.iter().position(|&j| j == idx).unwrap();
        eprintln!(
            "  {:<10} {:>4}/{:<5} {:>10.2e}  {:>11.4e} {:>11.4e}  {:>5}",
            t.name(),
            idxs.len(),
            ana.len(),
            err,
            ana[idx],
            num[at],
            idx
        );
        assert!(
            ana.iter().all(|v| v.is_finite()),
            "{} gradient contains NaN/Inf",
            t.name()
        );
        errs.push((t, err));
        if err > tol {
            failures.push(format!("{} (max rel err {:.3e} at idx {})", t.name(), err, idx));
        }
    }
    eprintln!();

    assert!(
        failures.is_empty(),
        "backward gradients disagree with finite differences for: {}",
        failures.join(", ")
    );
    errs
}

/// With DD-RoPE neutralised (theta = 0), every gradient must agree closely --
/// this exercises only the decay/trapezoidal recurrence path.
#[test]
fn ssm_scan_backward_matches_numerical_gradient_without_ddrope() {
    // 5% tolerance: kernels are built with -ffast-math, so f32 agreement
    // tighter than this is not expected.
    run_gradcheck(true, 5e-2);
}

#[test]
fn ssm_scan_backward_matches_numerical_gradient() {
    run_gradcheck(false, 5e-2);
}
