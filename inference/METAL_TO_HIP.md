# Porting `lumi-inference` from Metal to HIP (gfx1201 / RDNA4)

Target: AMD Radeon AI PRO R9700, gfx1201, wave32, 31.9 GiB VRAM, ROCm 7.14.

This document is a **plan and a trap list**, not a translation. The line-by-line
MSL→HIP rewrite is mechanical and left to you. What follows is the part that
*isn't* mechanical: the architectural decision that has to come first, the places
where a faithful translation would be wrong, and a staged order of work with a
verification gate at each step.

---

## Status

The native HIP path (**Option A** in §2) was chosen. Implementation is underway in the `inference/` crate.
Progress:
- Shared kernel crate (`kernels/`, with `lumi-kernels`) now exists at the workspace root, containing `csrc_hip/`, `csrc_cuda/`, and FFI layer (`ops.rs`).
- The Metal original is archived at `archive/metal-inference/` for reference.
- See `inference/README.md` for current status and CLI design.

---

## 1. Two findings that reshape the job

### 1.1 Candle has no AMD backend

`Cargo.toml` pulls `candle-core` with the `metal` feature. The available
backends are:

```
accelerate, cuda, cudarc, cudnn, mkl, metal
```

There is **no `rocm` or `hip` feature**, and none is planned upstream in the 0.9
line. So the Metal dependency is not confined to `ssm_step.metal` — it runs
through the whole tensor layer: `Linear`, `Embedding`, `matmul`, `softmax`,
`log_softmax`, every `Tensor` in `model.rs` and `eval.rs`.

Translating the `.metal` file alone leaves you with HIP kernels that have no way
to exchange buffers with the framework holding your weights.

### 1.2 You already own most of the kernels you need

`training/csrc_hip/` is a working, GPU-verified HIP kernel library, and its
`extern "C"` surface (see `training/src/native_ops.rs`) already exports nearly
every operation inference performs:

| Inference needs | Already in `training/csrc_hip/` |
|---|---|
| GEMV / GEMM | `matmul_f32`, `matmul_f32_bt`, hipBLAS handle mgmt |
| RMSNorm | `rmsnorm_fwd`, `rmsnorm_bias_fwd` |
| SiLU / sigmoid / gating | `silu_fwd`, `sigmoid_fwd`, `fused_silu_gate_fwd` |
| softplus, `-softplus` clamp | `softplus_fwd`, `neg_softplus_clamp` |
| Embedding lookup | `embedding_lookup` |
| in_proj slicing | `strided_split`, `fused_split_5` |
| Residual add | `elemwise_add` |
| Attention (if re-enabled) | `causal_softmax_fwd`, `gqa_expand`, `transpose_0213` |
| Device memory | `GpuBuf` in `training/src/gpu_memory.rs` |

Most of `ssm_step.metal` is **fusion of operations that already exist unfused in
HIP**. `pre_ssm_fused` fuses ~35 Candle dispatches; `rms_norm_fused`,
`gemv_fused`, and `gemv_residual_fused` all have direct unfused equivalents
above. Fusion was worth it on Metal because each Candle dispatch carried
launch overhead; you can re-earn that later, but you do not need it to be
*correct*.

**Genuinely new work is one kernel:** the single-token recurrent step that
carries `h` / `prev_bx` / `cum_angle` across calls (`ssm_step_fused`), plus its
windowed sibling (`ssm_window_fused`). Everything else is either already
written or is a fusion you can defer.

That reframes the task from "port 502 lines of MSL" to "write one recurrent
kernel and a host layer, reusing the training library."

---

## 2. Pick the architecture first

This decision gates everything else. You said native is on the table, which
matters, because it is the one that gets cleanest.

| Option | What it means | Verdict |
|---|---|---|
| **A. Native, mirroring `native_trainer.rs`** | Drop Candle. Link `training/csrc_hip/`, manage weights in `GpuBuf`, write the forward pass as explicit kernel calls. | **Recommended.** |
| B. Candle CPU + custom HIP kernels | Keep Candle on CPU for weights/matmul, HIP for the scan. | Constant host↔device copies per layer per token. Worst of both. |
| C. Port a ROCm backend to Candle | Weeks of work on someone else's abstraction. | Out of proportion. |
| D. Swap frameworks (burn+wgpu, ONNX Runtime) | wgpu has no hipBLAS-class GEMM; ORT means exporting the model. | Viable fallback, not first choice. |

### Why A

- The training binary is *already* a native HIP inference engine in the forward
  direction. `smoke_test_forward` runs embedding → RMSNorm → in_proj → SSM scan →
  cross-entropy with no framework at all.
- Weight loading already exists: `training/src/native_checkpoint.rs` loads
  safetensors straight into `GpuBuf`.
- One kernel library, one set of numerics, one place to fix bugs. Given that the
  training scan just turned out to have a real gradient bug that survived because
  two copies of similar logic drifted, a single shared kernel source is worth a
  lot.
- Inference is the *easy* direction — no backward pass, no optimizer state, no
  activation checkpointing. Fixed memory cost is weights only (~2 GiB fp32 at
  495M params), versus 7.39 GiB for training.

### The main cost of A

You rewrite `model.rs` (832 lines) and `eval.rs` (350 lines) against a raw
kernel API instead of `Tensor`. That is real work, but it is *ordinary* work —
no framework internals, no FFI archaeology. And `native_trainer.rs` is a
working reference for exactly this style, including the layer loop and buffer
reuse pattern.

### Structural suggestion

Move the kernels to a shared crate rather than duplicating them:

```
lil-mamba-3/
├── kernels/          ← new: csrc_hip/ + GpuBuf + native_ops FFI, one build.rs
├── training/         ← depends on kernels
└── inference/        ← depends on kernels
```

If that refactor feels like too much up front, have `inference/build.rs` compile
`../training/csrc_hip/*.cu` directly. Duplicating the `.cu` files is the one
thing to avoid — that is precisely the drift that hid the `theta` bug.

---

## 3. Staged plan

Each stage ends with something you can run. Do not proceed past a failing gate.

### Stage 0 — Capture golden vectors *first*

**Before touching any code.** You need a reference to compare against, and the
Mac may not be available later.

- If you still have the Mac: dump inputs and outputs of each Metal kernel for a
  fixed prompt to `.bin` files — `projected` in, all eight `pre_ssm_fused`
  outputs out; `x_heads`/`b_expanded`/`c_expanded`/`a_bar`/`lambda`/`h`/`prev_bx`
  in, `y_heads` + mutated state out. Include the state buffers *both* before and
  after, since the scan kernels mutate in place.
- If you do not: build the reference from `Device::Cpu`. Candle's CPU backend
  runs the same `model.rs` today, and the `--cpu` flag already exists in
  `main.rs`. This is arguably the better reference anyway — it is the
  framework's own definition of correct, independent of any GPU.
- Also capture end-to-end: a fixed prompt with temperature 0, and the resulting
  token IDs plus final logits.

**Gate:** golden files exist and the CPU path reproduces them deterministically.

### Stage 1 — Standalone kernel harness, no Rust

Translate the kernels and drive them from a small C++ `main()` compiled with
`hipcc`, loading the Stage 0 `.bin` files.

This deliberately keeps Rust, Candle, and FFI out of the picture while you are
still debugging kernel semantics. It is the same approach that caught the
wave-mask trap on the training side within minutes.

Order: `rms_norm_fused` (simplest, one reduction) → `gemv_fused` →
`pre_ssm_fused` → `ssm_step_fused` → `ssm_window_fused` (hardest, carries state
across timesteps).

**Gate:** each kernel matches its golden output to < 1e-4 relative. Diverge from
that and you are debugging one kernel, not a whole pipeline.

### Stage 2 — Cross-check the scan against training

`training/csrc_hip/ssm_scan.cu`'s `ssm_scan_fwd_gpu` computes the same
recurrence as `ssm_window_fused`, batched over the sequence. Feed both the same
inputs and compare.

This is a strong, independent check you already own — and it is now trustworthy,
because the training scan has a gradcheck behind it
(`training/tests/ssm_gradcheck.rs`).

If they disagree, one of them is wrong, and finding out *now* is much cheaper
than after the host layer exists. Note the training scan expects
`d_state == 64` (see §5.6).

**Gate:** inference scan and training scan agree on the same inputs.

### Stage 3 — Host layer in Rust

Replace `metal_ssm.rs` with a HIP equivalent. This is where Metal's binding
model disappears entirely — see §4.2. Reuse `GpuBuf` rather than writing new
memory management.

Keep the pipeline/state structs (`SsmStepPipeline`, `MetalSsmState`,
`PreSsmBuffers`) as shapes even though HIP has no "pipeline" concept; they are
good places to hang persistent device buffers. `MetalSsmState` becomes plain
`GpuBuf`s for `h`, `prev_bx`, `cum_angle`.

**Gate:** Rust calls each kernel and reproduces the Stage 1 results.

### Stage 4 — Model forward pass

Rewrite `model.rs` against the kernel API. Work outward from the innermost loop:
one Mamba block's `forward_step`, then the layer loop, then embedding and the
LM head.

Suggestion: get `forward_step` (single token) fully working before
`forward_window`. Generation only needs `forward_step`; `forward_window` is an
eval/perplexity optimization.

**Gate:** logits for a single token match the CPU reference.

### Stage 5 — End-to-end

`main.rs` generation loop, then `eval.rs`.

**Gate:** greedy generation from a fixed prompt produces the same token IDs as
the CPU reference. Expect exact match for the first several tokens, then
possible divergence as tiny float differences cross a sampling boundary — which
is why temperature 0 and comparing *logits* matters more than comparing text.

### Stage 6 — Performance

Only now. See §6.

---

## 4. Mapping reference

### 4.1 Language constructs

| Metal (MSL) | HIP | Note |
|---|---|---|
| `kernel void f(...)` | `__global__ void f(...)` | |
| `device const float*` | `const float* __restrict__` | |
| `device float*` | `float* __restrict__` | |
| `constant T&` | pass `T` by value | |
| `threadgroup float a[N]` | `__shared__ float a[N]` | |
| `threadgroup_barrier(mem_flags::mem_threadgroup)` | `__syncthreads()` | see §5.1 |
| `[[threadgroup_position_in_grid]]` | `blockIdx.x` | |
| `[[thread_position_in_threadgroup]]` | `threadIdx.x` | |
| `[[thread_index_in_simdgroup]]` | `threadIdx.x % 32` | wave32 |
| `[[simdgroup_index_in_threadgroup]]` | `threadIdx.x / 32` | wave32 |
| `simd_sum(v)` | shuffle-xor loop | **see §5.2 — not a one-liner** |
| `fast::exp` | `__expf` | |
| `fast::tanh` | `tanhf` | no `__tanhf`; see §5.3 |
| `fast::cos` / `fast::sin` | `__cosf` / `__sinf` | matches the training port |
| `rsqrt` | `rsqrtf` | |
| `clamp` | `fminf`/`fmaxf` or `fminf(fmaxf(...))` | |
| `fmod` | `fmodf` | |
| `M_PI_F` | define your own `3.14159265f` | HIP's `M_PI` is double |
| threadgroup | block / workgroup | vocabulary only |
| SIMD group | wave (32 lanes on RDNA) | sizes match |

### 4.2 Dispatch: the real structural change

Metal binds buffers to numbered slots, then dispatches:

```
encoder.set_buffer(0, Some(&proj_buf), 0);   // ... 14 of these
encoder.set_bytes_directly(14, size, &params);
encoder.dispatch_thread_groups(threadgroups, threads_per_group);
```

HIP has no binding table. Kernel arguments are just function arguments —
pointers and the params struct pass directly in the launch. The entire
`set_buffer` sequence in `metal_ssm.rs` collapses into the argument list of one
`<<<grid, block>>>` call.

`MTLSize { width: n_heads }` threadgroups → `grid = n_heads`.
`MTLSize { width: head_dim }` threads → `block = head_dim`.
Direct correspondence, no reinterpretation.

Follow the training pattern: keep kernels in `.cu`, expose one `extern "C"`
launcher per kernel that takes raw pointers and ints, and declare that in Rust.
Do not try to launch kernels from Rust directly.

Also gone: `metal_buffer(&tensor)` extracting a `Buffer` from a Candle tensor.
Under option A there is no tensor — you hold `GpuBuf` and pass `.ptr`.

---

## 5. Traps specific to this code

These are the places a faithful translation is wrong or fragile. Ranked by how
much time they will cost if missed.

### 5.1 Barrier divergence after early return — *highest risk*

`ssm_step_fused:235` and `ssm_window_fused:346`:

```
if (head_id >= (uint)n_heads || p >= (uint)head_dim) return;
```

then `threadgroup_barrier` at lines 275, 391, 416.

A thread that returns early **never reaches the barrier**. In HIP,
`__syncthreads()` where some threads of the block have exited is undefined —
it can hang the GPU or produce garbage, and on AMD it is far less forgiving than
on Apple silicon.

Today this is benign *only because* the dispatch uses `block = head_dim` exactly
(`metal_ssm.rs:418`, `:506`), so the guard never fires. That is an invariant
nobody has written down.

Do not carry the guard across as-is. Either drop the `p >= head_dim` clause and
assert `blockDim.x == head_dim` at launch, or restructure so all threads reach
every barrier and only the *work* is predicated. The same class of bug is what
made the training backward kernel trap on AMD.

`ssm_window_fused` is the dangerous one: its barriers are inside a `seq_len`
loop, so a mismatch hangs rather than merely corrupting.

### 5.2 `simd_sum` has no HIP equivalent

Metal's `simd_sum` is a full-wave reduction with no mask argument. HIP's
nearest equivalent is a shuffle-xor loop, and it comes with a constraint Metal
does not have:

- HIP's `__shfl_xor_sync` requires a **64-bit** mask (narrowed to 32 bits on
  wave32) — a 32-bit literal is a compile error.
- HIP asserts `mask == __ballot(true)`, i.e. the mask must name exactly the
  active lanes. Naming all 32 lanes inside a partially-divergent region
  **traps** with `HSA_STATUS_ERROR_EXCEPTION`.

Check every `simd_sum` call site for whether the full wave is converged there:

- `rms_norm_fused:31` — block is 32 threads, all active. Fine.
- `gemv_fused:465`, `gemv_residual_fused:497` — inside `if (row >= out_dim)
  return;`. **The last threadgroup can be partial.** If `out_dim` is not a
  multiple of 8, some SIMD groups exit and the surviving ones still hold a full
  wave — but verify per-lane, not per-group.
- `pre_ssm_fused:129`, `:143` — inside `if (simd_gid < n_groups)`. Divergence is
  at *SIMD-group* granularity, so each participating wave is internally
  converged. This one is fine, but it is fine by accident of the layout, not by
  construction.

`training/csrc_hip/ssm_scan.cu` has a working `block_reduce_sum` and a
`FULL_WARP_MASK` define — read those first; they encode the answers.

### 5.3 `fast::tanh` in the DD-RoPE path

`ssm_step_fused:255` and `ssm_window_fused:372` use `fast::tanh` to bound theta.
HIP has no `__tanhf`; use `tanhf`. This is *more* accurate than the Metal
original, not less, so it is safe — but note that the training kernels also use
`tanhf` here, so this keeps the two consistent. Consistency matters if you use
Stage 2 as a cross-check.

### 5.4 Unified memory assumptions are false on discrete VRAM

`gemv_fused:458`:

> read x directly from device memory (unified memory means x is likely in L2
> after first threadgroup reads it)

Apple silicon has unified memory and a shared last-level cache. The R9700 is a
**discrete** card: `x` sits in VRAM and every threadgroup re-reads it across the
memory bus. The optimization this comment describes does not exist on your
hardware.

This is a *semantic* difference, not a syntax one — the translated kernel will
be correct but slower than it looks. Staging `x` into `__shared__` once per
block is the AMD-appropriate structure. Defer to Stage 6, but do not carry the
comment across unexamined.

Related: host↔device transfers are real now. Anything relying on CPU and GPU
seeing the same pointer must become an explicit `hipMemcpy`.

### 5.5 Single-workgroup kernels waste the GPU

`pre_ssm_fused` dispatches **1 threadgroup of 1024 threads**
(`metal_ssm.rs:209`); `rms_norm_fused` dispatches **1 threadgroup of 32**
(`:284`). One workgroup occupies one compute unit. The R9700 has many.

On Apple this was reasonable — single-token decode is latency-bound and the GPU
is small. On a discrete AMD card it leaves nearly the whole device idle. It is
*correct*, so it is not a Stage 1 problem, but it is the first thing to look at
in Stage 6, and it may be simpler to just call the existing unfused training
kernels (which are already gridded sensibly) than to port these two at all.

Note also that `block = 32` is one wave. Metal's threadgroup barriers within a
single SIMD group are nearly free and the wave runs in lockstep; AMD wave32
behaves similarly, **but do not rely on implicit lockstep** — the compiler may
reorder without an explicit `__syncthreads()`.

### 5.6 `d_state == 64` is load-bearing

Two independent reasons:

- `ssm_window_fused:350` declares `float h_local[64]` / `prev_local[64]` with the
  comment "d_state ≤ 64 in current configs". Static private arrays; exceeding
  them is silent corruption.
- `ssm_step_fused:247` and `ssm_window_fused:361` declare
  `threadgroup float bc_shared[128]` = 2 × d_state with max d_state 64.
- The *training* backward kernel additionally requires `d_state == 64` exactly,
  because its DD-RoPE reduction assumes `d_state/2 == 32 ==` wave width. There is
  now an explicit guard for this in `ssm_scan.cu`.

Add the equivalent guard to the inference launchers. On AMD these assumptions
fail loudly (trap); on NVIDIA and Metal they fail silently, which is worse.

### 5.7 Private-array pressure

`ssm_window_fused` holds `h_local[64] + prev_local[64]` = 512 B/thread in
registers-or-scratch. With `block = 32` that is 16 KB/block, which is fine, but
watch the compiler's `private_seg_size` in the build output — the training scan
ended up at ~2 KB/thread of scratch from a `[512]` array, and scratch traffic on
AMD is slow. `hipcc -Rpass-analysis=kernel-resource-usage` will tell you.

### 5.8 In-place state mutation and launch ordering

`h`, `prev_bx`, and `cum_angle` are read-modify-write across kernel launches and
across timesteps. Metal's command encoder serializes within a queue. HIP kernels
on the same (default) stream also serialize, so the ordering carries over — but
if you ever add streams for overlap, these buffers are the hazard. Keep every
scan launch on one stream until Stage 6.

Also note `ssm_window_fused` writes `cum_angle` to *global* memory from thread
`p == 0` (line 375) while other threads read `b_rot`/`c_rot` from shared. The
barrier at 391 covers the shared-memory hazard; the global write is only safe
because one thread owns it. Preserve that ownership exactly.

### 5.9 Params structs and ABI

`SsmStepParams` etc. are `int`-only (plus one `float` in `RmsNormParams` /
`PreSsmParams`). Passing by value into a HIP kernel is fine, but if you go
through an `extern "C"` launcher, make the Rust `#[repr(C)]` struct match
exactly. Mismatched padding here produces plausible-looking wrong numbers rather
than a crash — the worst failure mode. Prefer passing plain scalars through the
launcher signature instead of a struct; the training FFI does this and it
sidesteps the whole question.

---

## 6. Performance, once it is correct

In rough order of expected value:

1. **Batch the decode.** Single-token inference on a discrete GPU is dominated
   by launch overhead and memory latency. The R9700 has far more parallelism
   than one token can use.
2. **Re-grid the single-workgroup kernels** (§5.5), or drop them for the
   existing training kernels.
3. **Stage GEMV inputs into LDS** (§5.4).
4. **Use hipBLAS for the projections.** `matmul_f32` already wraps
   `hipblasSgemm`. A hand-written GEMV is unlikely to beat it, and at batch > 1
   it becomes a GEMM where hipBLAS wins decisively.
5. **BF16 weights.** RDNA4 has no TF32, so fp32 GEMM runs on the vector ALUs
   with no matrix-core acceleration. The WMMA matrix cores are reached through
   BF16-in/FP32-accumulate — `matmul_bf16_from_f32*` already does this. This is
   the largest single speedup available and it halves weight memory too.
6. Only then consider re-fusing `pre_ssm_fused`.

---

## 7. Open decisions

1. **Shared kernel crate, or `inference/build.rs` reaching into
   `../training/csrc_hip/`?** Either works; copying the `.cu` files does not.
2. **Keep `forward_window` / `eval.rs` at all?** If perplexity eval is not on the
   critical path, Stage 5 gets much smaller — `ssm_window_fused` is the hardest
   kernel here.
3. **Attention layers.** `model.rs` has full attention/GQA/MLP support, but the
   training default is `attention_interval: 0`, so no checkpoint you train today
   will have attention weights. If you are not training hybrids, that whole path
   can be dropped rather than ported.
4. **Do you still have Mac access?** Determines whether Stage 0 golden vectors
   come from Metal or from Candle CPU. CPU is the better reference regardless;
   Metal is only useful for confirming the *existing* kernels were right.
5. **fp32 or bf16 weights on disk?** Affects the checkpoint loader you reuse from
   `native_checkpoint.rs`.

---

## 8. Suggested first session

1. `cargo run -p lumi-inference -- --cpu` with a fixed prompt, temperature 0.
   Confirm it still runs and capture the output tokens and logits. (Stage 0.)
2. Add a debug dump of `pre_ssm_fused`'s inputs and outputs from the CPU path.
3. Write a `hipcc` harness that loads those and runs your translated
   `rms_norm_fused`. Match to 1e-4.

That gets you a working compile-run-verify loop against real data before any
architectural commitment — and if the loop feels good, Stages 1–2 are mostly
repetition of it.
