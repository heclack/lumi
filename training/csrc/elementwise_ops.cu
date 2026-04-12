/*
 * Element-wise CUDA kernels for native training.
 * RMSNorm, SiLU, cross-entropy, vector add/mul, embedding lookup, AdamW.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

/* exp2f(x * LOG2E) is a single PTX instruction vs expf's range reduction */
#define LOG2E 1.44269504089f

extern "C" {

/* ─── RMSNorm Forward ─────────────────────────────────────────── */
/* out[i] = x[i] * rsqrt(mean(x[row]²) + eps) * gamma[col] */

__global__ void rmsnorm_fwd_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ gamma,
    float eps, int dim
) {
    int row = blockIdx.x;
    const float* x_row = x + row * dim;
    float* out_row = out + row * dim;

    // Compute mean(x²) using shared memory reduction
    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x_row[i];
        local_sum += val * val;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / dim + eps);

    // Apply normalization + scale
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = x_row[i] * rms * gamma[i];
    }
}

void rmsnorm_fwd(const float* x, float* out, const float* gamma,
                 float eps, int rows, int dim) {
    int threads = min(dim, 1024);
    rmsnorm_fwd_kernel<<<rows, threads, threads * sizeof(float)>>>(
        x, out, gamma, eps, dim);
}

/* ─── Fused RMSNorm + Bias Forward ─────────────────────────── */
/* out[i] = x[i] * rms * gamma[i] + bias[i] */

__global__ void rmsnorm_bias_fwd_kernel(
    const float* __restrict__ x, float* __restrict__ out,
    const float* __restrict__ gamma, const float* __restrict__ bias,
    float eps, int dim
) {
    int row = blockIdx.x;
    const float* x_row = x + row * dim;
    float* out_row = out + row * dim;
    extern __shared__ float shared[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x_row[i];
        local_sum += val * val;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float rms = rsqrtf(shared[0] / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = x_row[i] * rms * gamma[i] + bias[i];
    }
}

void rmsnorm_bias_fwd(const float* x, float* out, const float* gamma,
                      const float* bias, float eps, int rows, int dim) {
    int threads = min(dim, 1024);
    rmsnorm_bias_fwd_kernel<<<rows, threads, threads * sizeof(float)>>>(
        x, out, gamma, bias, eps, dim);
}

/* ─── Fused RMSNorm + Bias Backward ────────────────────────── */
/* Same as rmsnorm_bwd but also accumulates dbias[i] += sum_rows(dy[i]) */

__global__ void rmsnorm_bias_bwd_kernel(
    const float* __restrict__ x,
    const float* dy,              // may alias dx
    const float* __restrict__ gamma,
    float* dx,                    // may alias dy
    float* __restrict__ dgamma,
    float* __restrict__ dbias,
    float eps, int dim
) {
    int row = blockIdx.x;
    const float* x_row = x + row * dim;
    const float* dy_row = dy + row * dim;
    float* dx_row = dx + row * dim;
    extern __shared__ float shared[];

    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x_row[i];
        local_sq += val * val;
    }
    shared[threadIdx.x] = local_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float mean_sq = shared[0] / dim;
    float rms = rsqrtf(mean_sq + eps);

    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_dot += dy_row[i] * gamma[i] * x_row[i] * rms;
    }
    shared[threadIdx.x] = local_dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float dot = shared[0];

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float dy_val = dy_row[i];
        float x_hat = x_row[i] * rms;
        dx_row[i] = (dy_val * gamma[i] - x_hat * dot / dim) * rms;
        atomicAdd(&dgamma[i], dy_val * x_hat);
        atomicAdd(&dbias[i], dy_val);
    }
}

void rmsnorm_bias_bwd(const float* x, const float* dy, const float* gamma,
                      float* dx, float* dgamma, float* dbias,
                      float eps, int rows, int dim) {
    int threads = min(dim, 1024);
    rmsnorm_bias_bwd_kernel<<<rows, threads, threads * sizeof(float)>>>(
        x, dy, gamma, dx, dgamma, dbias, eps, dim);
}

/* ─── RMSNorm Backward ───────────────────────────────────────── */
/* dx = dy * gamma * rms - x * (sum(dy * gamma * x) / (dim * norm²)) * rms */
/* dgamma += sum_over_rows(dy * x * rms) */

__global__ void rmsnorm_bwd_kernel(
    const float* __restrict__ x,
    const float* dy,              // may alias dx
    const float* __restrict__ gamma,
    float* dx,                    // may alias dy
    float* __restrict__ dgamma,
    float eps, int dim
) {
    int row = blockIdx.x;
    const float* x_row = x + row * dim;
    const float* dy_row = dy + row * dim;
    float* dx_row = dx + row * dim;

    extern __shared__ float shared[];

    // Compute mean(x²) for this row
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = x_row[i];
        local_sq += val * val;
    }
    shared[threadIdx.x] = local_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float mean_sq = shared[0] / dim;
    float rms = rsqrtf(mean_sq + eps);

    // Compute dot(dy * gamma, x * rms) for this row
    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_dot += dy_row[i] * gamma[i] * x_row[i] * rms;
    }
    shared[threadIdx.x] = local_dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float dot_sum = shared[0];

    // dx = (dy * gamma - x * dot_sum / dim) * rms
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float dy_val = dy_row[i];
        float x_hat = x_row[i] * rms;
        dx_row[i] = (dy_val * gamma[i] - x_hat * dot_sum / dim) * rms;
        atomicAdd(&dgamma[i], dy_val * x_hat);
    }
}

void rmsnorm_bwd(const float* x, const float* dy, const float* gamma,
                 float* dx, float* dgamma, float eps, int rows, int dim) {
    int threads = min(dim, 1024);
    cudaMemset(dgamma, 0, dim * sizeof(float));
    rmsnorm_bwd_kernel<<<rows, threads, threads * sizeof(float)>>>(
        x, dy, gamma, dx, dgamma, eps, dim);
}

/* ─── SiLU Forward (float4 vectorized) ────────────────────────── */
/* y = x * sigmoid(x) */

__global__ void silu_fwd_kernel_vec4(const float4* x, float4* y, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = x[i];
        float4 out;
        out.x = v.x / (1.0f + exp2f(-v.x * LOG2E));
        out.y = v.y / (1.0f + exp2f(-v.y * LOG2E));
        out.z = v.z / (1.0f + exp2f(-v.z * LOG2E));
        out.w = v.w / (1.0f + exp2f(-v.w * LOG2E));
        y[i] = out;
    }
}

__global__ void silu_fwd_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        y[i] = val / (1.0f + exp2f(-val * LOG2E));
    }
}

void silu_fwd(const float* x, float* y, int n) {
    if (n % 4 == 0) {
        int n4 = n / 4;
        silu_fwd_kernel_vec4<<<(n4 + 255) / 256, 256>>>((const float4*)x, (float4*)y, n4);
    } else {
        silu_fwd_kernel<<<(n + 255) / 256, 256>>>(x, y, n);
    }
}

/* ─── SiLU Backward ───────────────────────────────────────────── */
/* dx = dy * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))) */
/*    = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x))) */

__global__ void silu_bwd_kernel_vec4(const float4* x, const float4* dy, float4* dx, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = x[i], dv = dy[i]; float4 out;
        float sx = 1.0f / (1.0f + exp2f(-v.x * LOG2E));
        float sy = 1.0f / (1.0f + exp2f(-v.y * LOG2E));
        float sz = 1.0f / (1.0f + exp2f(-v.z * LOG2E));
        float sw = 1.0f / (1.0f + exp2f(-v.w * LOG2E));
        out.x = dv.x * sx * (1.0f + v.x * (1.0f - sx));
        out.y = dv.y * sy * (1.0f + v.y * (1.0f - sy));
        out.z = dv.z * sz * (1.0f + v.z * (1.0f - sz));
        out.w = dv.w * sw * (1.0f + v.w * (1.0f - sw));
        dx[i] = out;
    }
}

__global__ void silu_bwd_kernel(const float* x, const float* dy, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        float sig = 1.0f / (1.0f + exp2f(-val * LOG2E));
        dx[i] = dy[i] * sig * (1.0f + val * (1.0f - sig));
    }
}

void silu_bwd(const float* x, const float* dy, float* dx, int n) {
    if (n % 4 == 0) {
        int n4 = n / 4;
        silu_bwd_kernel_vec4<<<(n4 + 255) / 256, 256>>>(
            (const float4*)x, (const float4*)dy, (float4*)dx, n4);
    } else {
        silu_bwd_kernel<<<(n + 255) / 256, 256>>>(x, dy, dx, n);
    }
}

/* ─── Sigmoid Forward (float4 vectorized) ─────────────────────── */
/* y = 1 / (1 + exp(-x)) */

__global__ void sigmoid_fwd_kernel_vec4(const float4* x, float4* y, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = x[i];
        float4 out;
        out.x = 1.0f / (1.0f + exp2f(-v.x * LOG2E));
        out.y = 1.0f / (1.0f + exp2f(-v.y * LOG2E));
        out.z = 1.0f / (1.0f + exp2f(-v.z * LOG2E));
        out.w = 1.0f / (1.0f + exp2f(-v.w * LOG2E));
        y[i] = out;
    }
}

__global__ void sigmoid_fwd_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = 1.0f / (1.0f + exp2f(-x[i] * LOG2E));
    }
}

void sigmoid_fwd(const float* x, float* y, int n) {
    if (n % 4 == 0) {
        int n4 = n / 4;
        sigmoid_fwd_kernel_vec4<<<(n4 + 255) / 256, 256>>>((const float4*)x, (float4*)y, n4);
    } else {
        sigmoid_fwd_kernel<<<(n + 255) / 256, 256>>>(x, y, n);
    }
}

/* ─── Sigmoid Backward ────────────────────────────────────────── */
/* dx = dy * sigmoid(x) * (1 - sigmoid(x)) */

__global__ void sigmoid_bwd_kernel(const float* x, const float* dy, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sig = 1.0f / (1.0f + exp2f(-x[i] * LOG2E));
        dx[i] = dy[i] * sig * (1.0f - sig);
    }
}

void sigmoid_bwd(const float* x, const float* dy, float* dx, int n) {
    sigmoid_bwd_kernel<<<(n + 255) / 256, 256>>>(x, dy, dx, n);
}

/* ─── Softplus Forward ────────────────────────────────────────── */
/* y = log(1 + exp(x)), with guard for large x */

__global__ void softplus_fwd_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        y[i] = (val > 20.0f) ? val : logf(1.0f + exp2f(val * LOG2E));
    }
}

void softplus_fwd(const float* x, float* y, int n) {
    softplus_fwd_kernel<<<(n + 255) / 256, 256>>>(x, y, n);
}

/* ─── Softplus Backward ──────────────────────────────────────── */
/* dx = dy * sigmoid(x)  (derivative of softplus is sigmoid) */

__global__ void softplus_bwd_kernel(const float* x, const float* dy, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sig = 1.0f / (1.0f + exp2f(-x[i] * LOG2E));
        dx[i] = dy[i] * sig;
    }
}

void softplus_bwd(const float* x, const float* dy, float* dx, int n) {
    softplus_bwd_kernel<<<(n + 255) / 256, 256>>>(x, dy, dx, n);
}

/* ─── Negate ─────────────────────────────────────────────────── */
/* y[i] = -x[i] */

__global__ void negate_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = -x[i];
    }
}

void negate(const float* x, float* y, int n) {
    negate_kernel<<<(n + 255) / 256, 256>>>(x, y, n);
}

/* ─── Fused Neg-Softplus-Clamp (for data-dependent A) ────────── */
/* y = clamp(-softplus(x), min_val, max_val)
 * Replaces 3 separate kernel launches: softplus_fwd + negate + clamp_values */

__global__ void neg_softplus_clamp_kernel(const float* x, float* y, float min_val, float max_val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        float sp = (val > 20.0f) ? val : logf(1.0f + exp2f(val * LOG2E));
        float neg_sp = -sp;
        y[i] = fminf(fmaxf(neg_sp, min_val), max_val);
    }
}

void neg_softplus_clamp(const float* x, float* y, float min_val, float max_val, int n) {
    neg_softplus_clamp_kernel<<<(n + 255) / 256, 256>>>(x, y, min_val, max_val, n);
}

/* ─── Fused SiLU-Gate Forward (float4 vectorized) ────────────── */
/* y[i] = ssm_out[i] * z[i] * sigmoid(z[i])  (= ssm_out * SiLU(z)) */

__global__ void fused_silu_gate_fwd_kernel_vec4(
    const float4* ssm_out, const float4* z, float4* y, int n4
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 s = ssm_out[i], v = z[i]; float4 out;
        out.x = s.x * v.x / (1.0f + exp2f(-v.x * LOG2E));
        out.y = s.y * v.y / (1.0f + exp2f(-v.y * LOG2E));
        out.z = s.z * v.z / (1.0f + exp2f(-v.z * LOG2E));
        out.w = s.w * v.w / (1.0f + exp2f(-v.w * LOG2E));
        y[i] = out;
    }
}

__global__ void fused_silu_gate_fwd_kernel(
    const float* ssm_out, const float* z, float* y, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sig = 1.0f / (1.0f + exp2f(-z[i] * LOG2E));
        y[i] = ssm_out[i] * z[i] * sig;
    }
}

void fused_silu_gate_fwd(const float* ssm_out, const float* z, float* y, int n) {
    if (n % 4 == 0) {
        int n4 = n / 4;
        fused_silu_gate_fwd_kernel_vec4<<<(n4 + 255) / 256, 256>>>(
            (const float4*)ssm_out, (const float4*)z, (float4*)y, n4);
    } else {
        fused_silu_gate_fwd_kernel<<<(n + 255) / 256, 256>>>(ssm_out, z, y, n);
    }
}

/* ��── Fused SiLU-Gate Backward ───────────────────────────────── */
/* d_ssm_out[i] = dy[i] * silu(z[i])
 * d_z[i] = dy[i] * ssm_out[i] * sigmoid(z[i]) * (1 + z[i] * (1 - sigmoid(z[i])))
 * Uses raw z (before SiLU) and ssm_out. */

__global__ void fused_silu_gate_bwd_kernel(
    const float* ssm_out, const float* z, const float* dy,
    float* d_ssm_out, float* d_z, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = ssm_out[i], v = z[i], g = dy[i];
        float sig = 1.0f / (1.0f + exp2f(-v * LOG2E));
        float silu_z = v * sig;
        d_ssm_out[i] = g * silu_z;
        d_z[i] = g * s * sig * (1.0f + v * (1.0f - sig));
    }
}

void fused_silu_gate_bwd(const float* ssm_out, const float* z, const float* dy,
                          float* d_ssm_out, float* d_z, int n) {
    fused_silu_gate_bwd_kernel<<<(n + 255) / 256, 256>>>(ssm_out, z, dy, d_ssm_out, d_z, n);
}

/* ─── Element-wise Multiply ───────────────────────────────────── */
/* y = a * b */

__global__ void elemwise_mul_kernel_vec4(const float4* a, const float4* b, float4* y, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 va = a[i], vb = b[i];
        y[i] = make_float4(va.x*vb.x, va.y*vb.y, va.z*vb.z, va.w*vb.w);
    }
}

__global__ void elemwise_mul_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

void elemwise_mul(const float* a, const float* b, float* y, int n) {
    if (n % 4 == 0) {
        int n4 = n / 4;
        elemwise_mul_kernel_vec4<<<(n4 + 255) / 256, 256>>>((const float4*)a, (const float4*)b, (float4*)y, n4);
    } else {
        elemwise_mul_kernel<<<(n + 255) / 256, 256>>>(a, b, y, n);
    }
}

/* ─── Element-wise Add (float4 vectorized) ────────────────────── */
/* y = a + b */

__global__ void elemwise_add_kernel_vec4(const float4* a, const float4* b, float4* y, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 va = a[i], vb = b[i];
        y[i] = make_float4(va.x+vb.x, va.y+vb.y, va.z+vb.z, va.w+vb.w);
    }
}

__global__ void elemwise_add_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

void elemwise_add(const float* a, const float* b, float* y, int n) {
    if (n % 4 == 0) {
        int n4 = n / 4;
        elemwise_add_kernel_vec4<<<(n4 + 255) / 256, 256>>>((const float4*)a, (const float4*)b, (float4*)y, n4);
    } else {
        elemwise_add_kernel<<<(n + 255) / 256, 256>>>(a, b, y, n);
    }
}

/* ─── Broadcast Bias Add: out[i*dim+j] = x[i*dim+j] + bias[j] ── */

__global__ void bias_add_kernel(const float* x, const float* bias, float* y, int rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * dim) {
        y[idx] = x[idx] + bias[idx % dim];
    }
}

void bias_add_fwd(const float* x, const float* bias, float* y, int rows, int dim) {
    int n = rows * dim;
    bias_add_kernel<<<(n + 255) / 256, 256>>>(x, bias, y, rows, dim);
}

/* Bias backward: d_bias[j] = sum_i(dy[i*dim+j])
 * Row-parallel: each block handles a group of rows, threads collaborate on dim.
 * Coalesced reads (threads read consecutive dim elements in the same row). */
__global__ void bias_add_bwd_kernel(const float* dy, float* d_bias, int rows, int dim) {
    int tid = threadIdx.x;
    int rows_per_block = 32;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, rows);

    for (int j = tid; j < dim; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = row_start; i < row_end; i++) {
            sum += dy[i * dim + j];
        }
        atomicAdd(&d_bias[j], sum);
    }
}

void bias_add_bwd(const float* dy, float* d_bias, int rows, int dim) {
    int rows_per_block = 32;
    int n_blocks = (rows + rows_per_block - 1) / rows_per_block;
    int threads = min(dim, 256);
    bias_add_bwd_kernel<<<n_blocks, threads>>>(dy, d_bias, rows, dim);
}

/* ─── Element-wise Multiply Backward ──────────────────────────── */
/* Given y = a * b, dy: da = dy * b, db = dy * a */

__global__ void elemwise_mul_bwd_kernel(
    const float* a, const float* b, const float* dy,
    float* da, float* db, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        da[i] = dy[i] * b[i];
        db[i] = dy[i] * a[i];
    }
}

void elemwise_mul_bwd(const float* a, const float* b, const float* dy,
                      float* da, float* db, int n) {
    elemwise_mul_bwd_kernel<<<(n + 255) / 256, 256>>>(a, b, dy, da, db, n);
}

/* ─── Embedding Lookup ────────────────────────────────────────── */
/* out[i] = embedding[token_ids[i], :] */

__global__ void embedding_lookup_kernel(
    const float* embedding, const int* token_ids,
    float* out, int seq_len, int dim
) {
    int pos = blockIdx.x;
    int tid = token_ids[pos];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[pos * dim + i] = embedding[tid * dim + i];
    }
}

void embedding_lookup(const float* embedding, const int* token_ids,
                      float* out, int batch_seq, int dim) {
    embedding_lookup_kernel<<<batch_seq, min(dim, 1024)>>>(
        embedding, token_ids, out, batch_seq, dim);
}

/* ─── Sparse Cross-Entropy Forward ────────────────────────────── */
/* loss = -mean(log_softmax(logits)[targets]) */

__global__ void sparse_ce_fwd_kernel(
    const float* logits, const int* targets,
    float* per_token_loss, int vocab
) {
    int pos = blockIdx.x;
    const float* row = logits + pos * vocab;
    int target = targets[pos];

    // Bounds check: invalid target → zero loss
    if (target < 0 || target >= vocab) {
        if (threadIdx.x == 0) per_token_loss[pos] = 0.0f;
        return;
    }

    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }
    // Warp reduction for max
    extern __shared__ float shared[];
    shared[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = shared[0];

    // Compute log(sum(exp(x - max)))
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        local_sum += exp2f((row[i] - max_val) * LOG2E);
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float log_sum_exp = logf(shared[0]) + max_val;

    // loss = -(logits[target] - log_sum_exp)
    if (threadIdx.x == 0) {
        per_token_loss[pos] = -(row[target] - log_sum_exp);
    }
}

/* Reduce per-token losses to scalar mean.
 *
 * TODO: this tree reduction `for (s = blockDim.x/2; s > 0; s >>= 1)` silently drops
 * elements when blockDim.x is not a power of 2. The launch site below uses
 * `min(batch_seq, 1024)` threads, so when batch_seq < 1024 and not a power of 2 the
 * resulting scalar is wrong. The Rust trainer reads this scalar (`buf.loss`) for the
 * displayed/checkpointed training loss, so a bad config here corrupts logging — the
 * optimizer is unaffected because gradients flow through sparse_ce_bwd_kernel, not
 * this reduction.
 *
 * Separately, this kernel has been observed to produce slightly inaccurate results on
 * sm_120 (RTX 5090 / Blackwell consumer) — the validation path in native_trainer.rs
 * works around it by reducing per_token_loss on the CPU instead.
 *
 * Fix is to either (a) pad blockDim.x to a power of 2 and gate accumulation, or
 * (b) two-pass reduction with atomicAdd. Until then, the Rust trainer warns at
 * startup if batch*seq trips the power-of-2 case.
 */
__global__ void reduce_mean_kernel(const float* values, float* out, int n) {
    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += values[i];
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = shared[0] / n;
    }
}

void sparse_cross_entropy_fwd(const float* logits, const int* targets,
                               float* loss, float* per_token_loss,
                               int batch_seq, int vocab) {
    int threads = min(vocab, 1024);
    sparse_ce_fwd_kernel<<<batch_seq, threads, threads * sizeof(float)>>>(
        logits, targets, per_token_loss, vocab);
    reduce_mean_kernel<<<1, min(batch_seq, 1024), min(batch_seq, 1024) * sizeof(float)>>>(
        per_token_loss, loss, batch_seq);
}

/* ─── Sparse Cross-Entropy Backward ───────────────────────────── */
/* d_logits = softmax(logits) - one_hot(targets), scaled by 1/batch_seq */

__global__ void sparse_ce_bwd_kernel(
    const float* logits, const int* targets,
    float* d_logits, int vocab, float scale
) {
    int pos = blockIdx.x;
    const float* row = logits + pos * vocab;
    float* drow = d_logits + pos * vocab;
    int target = targets[pos];

    // Find max
    extern __shared__ float shared[];
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x)
        max_val = fmaxf(max_val, row[i]);
    shared[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = shared[0];

    // Compute sum(exp)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x)
        local_sum += exp2f((row[i] - max_val) * LOG2E);
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float sum_exp = shared[0];

    // d_logits = (softmax - one_hot) * scale
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float softmax_val = exp2f((row[i] - max_val) * LOG2E) / sum_exp;
        float one_hot = (i == target) ? 1.0f : 0.0f;
        drow[i] = (softmax_val - one_hot) * scale;
    }
}

void sparse_cross_entropy_bwd(const float* logits, const int* targets,
                               float* d_logits, int batch_seq, int vocab) {
    int threads = min(vocab, 1024);
    float scale = 1.0f / batch_seq;
    sparse_ce_bwd_kernel<<<batch_seq, threads, threads * sizeof(float)>>>(
        logits, targets, d_logits, vocab, scale);
}

/* ─── AdamW Optimizer ─────────────────────────────────────────── */
/* Fused: m update + v update + weight decay + param update in one kernel */

__global__ void adamw_kernel(
    float* __restrict__ w,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;

        float m_hat = mi / bias_correction1;
        float v_hat = vi / bias_correction2;

        w[i] = w[i] * (1.0f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void adamw_update(float* w, const float* grad, float* m, float* v,
                  float lr, float beta1, float beta2, float eps,
                  float weight_decay, int step, int n) {
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);
    adamw_kernel<<<(n + 255) / 256, 256>>>(
        w, grad, m, v, lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2, n);
}

/* ─── Fused AdamW + Gradient Clipping ─────────────────────────────── */
/* Reads total_norm_sq from device, computes clip scale, applies to grad inline.
 * Eliminates: D2H for grad norm, separate grad_scale kernel launches. */

__global__ void adamw_clipped_kernel(
    float* __restrict__ w,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ total_norm_sq_ptr,
    float max_norm,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2,
    int n
) {
    // Thread 0 of each block computes clip_scale from global grad norm
    __shared__ float clip_scale;
    if (threadIdx.x == 0) {
        float total_norm = sqrtf(*total_norm_sq_ptr);
        if (total_norm != total_norm || total_norm > 1e30f) {
            clip_scale = 0.0f; // NaN/inf — skip update
        } else if (total_norm > max_norm) {
            clip_scale = max_norm / (total_norm + 1e-6f);
        } else {
            clip_scale = 1.0f;
        }
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i] * clip_scale;
        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;

        float m_hat = mi / bias_correction1;
        float v_hat = vi / bias_correction2;

        w[i] = w[i] * (1.0f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void adamw_clipped_update(float* w, const float* grad, float* m, float* v,
                          const float* total_norm_sq_ptr, float max_norm,
                          float lr, float beta1, float beta2, float eps,
                          float weight_decay, int step, int n) {
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);
    adamw_clipped_kernel<<<(n + 255) / 256, 256>>>(
        w, grad, m, v, total_norm_sq_ptr, max_norm,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2, n);
}

/* ─── Fused 5-way Split (extract all 5 column ranges in one kernel) ─ */
/* Splits [rows, total_cols] into 5 contiguous output buffers at once. */

__global__ void fused_split_5_kernel(
    const float* src,
    float* dst0, float* dst1, float* dst2, float* dst3, float* dst4,
    int rows, int total_cols,
    int off0, int len0,
    int off1, int len1,
    int off2, int len2,
    int off3, int len3,
    int off4, int len4
) {
    int row = blockIdx.x;
    const float* src_row = src + row * total_cols;
    for (int c = threadIdx.x; c < len0; c += blockDim.x)
        dst0[row * len0 + c] = src_row[off0 + c];
    for (int c = threadIdx.x; c < len1; c += blockDim.x)
        dst1[row * len1 + c] = src_row[off1 + c];
    for (int c = threadIdx.x; c < len2; c += blockDim.x)
        dst2[row * len2 + c] = src_row[off2 + c];
    for (int c = threadIdx.x; c < len3; c += blockDim.x)
        dst3[row * len3 + c] = src_row[off3 + c];
    for (int c = threadIdx.x; c < len4; c += blockDim.x)
        dst4[row * len4 + c] = src_row[off4 + c];
}

void fused_split_5(
    const float* src,
    float* dst0, float* dst1, float* dst2, float* dst3, float* dst4,
    int rows, int total_cols,
    int off0, int len0,
    int off1, int len1,
    int off2, int len2,
    int off3, int len3,
    int off4, int len4
) {
    int max_len = max(max(max(len0, len1), max(len2, len3)), len4);
    fused_split_5_kernel<<<rows, min(max_len, 1024)>>>(
        src, dst0, dst1, dst2, dst3, dst4,
        rows, total_cols, off0, len0, off1, len1, off2, len2, off3, len3, off4, len4);
}

/* ─── Strided Split (extract column range from row-major matrix) ─ */
/* src: [rows, total_cols], dst: [rows, split_cols] */
/* Copies columns [col_offset..col_offset+split_cols] from each row */

__global__ void strided_split_kernel(
    const float* src, float* dst,
    int rows, int total_cols, int col_offset, int split_cols
) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    for (int c = col; c < split_cols; c += blockDim.x) {
        dst[row * split_cols + c] = src[row * total_cols + col_offset + c];
    }
}

void strided_split(const float* src, float* dst,
                   int rows, int total_cols, int col_offset, int split_cols) {
    strided_split_kernel<<<rows, min(split_cols, 1024)>>>(
        src, dst, rows, total_cols, col_offset, split_cols);
}

/* ─── Strided Assemble (write column range into row-major matrix) ── */
/* Reverse of strided_split: writes dst[row, col_offset..col_offset+split_cols] from src */

__global__ void strided_assemble_kernel(
    const float* src, float* dst,
    int rows, int total_cols, int col_offset, int split_cols
) {
    int row = blockIdx.x;
    for (int c = threadIdx.x; c < split_cols; c += blockDim.x) {
        dst[row * total_cols + col_offset + c] = src[row * split_cols + c];
    }
}

void strided_assemble(const float* src, float* dst,
                      int rows, int total_cols, int col_offset, int split_cols) {
    strided_assemble_kernel<<<rows, min(split_cols, 1024)>>>(
        src, dst, rows, total_cols, col_offset, split_cols);
}

/* ─── Embedding Backward ──────────────────────────────────────── */
/* Scatter-add dy into d_embedding rows indexed by token_ids.    */
/* d_embedding[token_ids[pos], :] += dy[pos, :]                  */

__global__ void embedding_bwd_kernel(
    const float* __restrict__ dy,
    const int* __restrict__ token_ids,
    float* __restrict__ d_embedding,
    int batch_seq, int dim, int vocab
) {
    int pos = blockIdx.x;
    if (pos >= batch_seq) return;
    int tid = token_ids[pos];
    if (tid < 0 || tid >= vocab) return;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        atomicAdd(&d_embedding[tid * dim + i], dy[pos * dim + i]);
    }
}

void embedding_bwd(const float* dy, const int* token_ids,
                   float* d_embedding, int batch_seq, int dim, int vocab) {
    embedding_bwd_kernel<<<batch_seq, min(dim, 1024)>>>(
        dy, token_ids, d_embedding, batch_seq, dim, vocab);
}

/* ─── Gradient clipping kernels ────────────────────────────────── */

__global__ void grad_norm_squared_kernel(const float* __restrict__ buf, float* __restrict__ out, int n) {
    __shared__ float partial[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float v = buf[i];
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, partial[0]);
}

void grad_norm_squared(const float* buf, float* output, int n) {
    // NOTE: caller must zero `output` before the accumulation loop.
    // Do NOT add cudaMemset here — it's synchronous and causes a full pipeline stall.
    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 1024);
    grad_norm_squared_kernel<<<blocks, threads>>>(buf, output, n);
}

__global__ void grad_scale_kernel(float* __restrict__ buf, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) buf[idx] *= scale;
}

void grad_scale(float* buf, float scale, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    grad_scale_kernel<<<blocks, threads>>>(buf, scale, n);
}

__global__ void clamp_values_kernel(float* __restrict__ buf, float min_val, float max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        buf[idx] = fminf(fmaxf(buf[idx], min_val), max_val);
    }
}

void clamp_values(float* buf, float min_val, float max_val, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clamp_values_kernel<<<blocks, threads>>>(buf, min_val, max_val, n);
}

/* ─── Causal Softmax Forward ──────────────────────────────────── */
/* scores: [batch_heads, seq, seq] → output: [batch_heads, seq, seq]
 * For each row: mask future positions (col > row) with -inf,
 * then compute stable softmax: exp(x - max) / sum(exp(x - max))
 * One block per (batch_head, row), threads collaborate on columns.
 */

__global__ void causal_softmax_fwd_kernel(
    const float* __restrict__ scores,
    float* __restrict__ output,
    int seq, int window_size
) {
    int bh = blockIdx.x;   // which (batch, head) pair
    int row = blockIdx.y;  // which query position
    int tid = threadIdx.x;
    int base = bh * seq * seq + row * seq;
    // Sliding window: start_col = max(0, row - window_size + 1), or 0 if window_size=0 (full causal)
    int start_col = (window_size > 0) ? max(0, row - window_size + 1) : 0;

    extern __shared__ float sdata[];

    // Step 1: Find max (window: cols start_col..row)
    float max_val = -1e30f;
    for (int col = start_col + tid; col <= row; col += blockDim.x) {
        float v = scores[base + col];
        if (v > max_val) max_val = v;
    }
    sdata[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    max_val = sdata[0];

    // Step 2: Compute exp(x - max) and sum (window: cols start_col..row only)
    float sum_val = 0.0f;
    for (int col = start_col + tid; col <= row; col += blockDim.x) {
        float e = exp2f((scores[base + col] - max_val) * LOG2E);
        output[base + col] = e;
        sum_val += e;
    }
    // Zero positions outside window (before start_col and after row)
    for (int col = tid; col < start_col; col += blockDim.x) {
        output[base + col] = 0.0f;
    }
    for (int col = tid + row + 1; col < seq; col += blockDim.x) {
        output[base + col] = 0.0f;
    }
    sdata[tid] = sum_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum_val = sdata[0];

    // Step 3: Normalize
    float inv_sum = (sum_val > 0.0f) ? 1.0f / sum_val : 0.0f;
    for (int col = start_col + tid; col <= row; col += blockDim.x) {
        output[base + col] *= inv_sum;
    }
}

void causal_softmax_fwd(const float* scores, float* output, int batch_heads, int seq, int window_size) {
    dim3 grid(batch_heads, seq);
    int threads = min(seq, 256);
    int shared = threads * sizeof(float);
    causal_softmax_fwd_kernel<<<grid, threads, shared>>>(scores, output, seq, window_size);
}

/* ─── Causal Softmax Backward ────────────────────────────────── */
/* d_scores = output * (d_output - sum(d_output * output))
 * Only for causal positions (col <= row).
 */

__global__ void causal_softmax_bwd_kernel(
    const float* __restrict__ output,
    const float* __restrict__ d_output,
    float* __restrict__ d_scores,
    int seq, int window_size
) {
    int bh = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int base = bh * seq * seq + row * seq;
    int start_col = (window_size > 0) ? max(0, row - window_size + 1) : 0;

    extern __shared__ float sdata[];

    // Compute dot = sum(d_output * output) for this row (window only)
    float dot = 0.0f;
    for (int col = start_col + tid; col <= row; col += blockDim.x) {
        dot += d_output[base + col] * output[base + col];
    }
    sdata[tid] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    dot = sdata[0];

    // d_scores[col] = output[col] * (d_output[col] - dot)
    for (int col = start_col + tid; col <= row; col += blockDim.x) {
        d_scores[base + col] = output[base + col] * (d_output[base + col] - dot);
    }
    // Zero positions outside window
    for (int col = tid; col < start_col; col += blockDim.x) {
        d_scores[base + col] = 0.0f;
    }
    for (int col = tid + row + 1; col < seq; col += blockDim.x) {
        d_scores[base + col] = 0.0f;
    }
}

void causal_softmax_bwd(const float* output, const float* d_output, float* d_scores,
                        int batch_heads, int seq, int window_size) {
    dim3 grid(batch_heads, seq);
    int threads = min(seq, 256);
    int shared = threads * sizeof(float);
    causal_softmax_bwd_kernel<<<grid, threads, shared>>>(output, d_output, d_scores, seq, window_size);
}

/* ─── GQA Expand: repeat KV heads to match query heads ────────── */
/* src: [batch, kv_heads, seq, head_dim] → dst: [batch, n_heads, seq, head_dim]
 * Each KV head is repeated (n_heads / kv_heads) times.
 */

__global__ void gqa_expand_kernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int batch, int kv_heads, int n_heads, int seq, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_heads * seq * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int s = (idx / head_dim) % seq;
    int h = (idx / (head_dim * seq)) % n_heads;
    int b = idx / (head_dim * seq * n_heads);

    int repeats = n_heads / kv_heads;
    int kv_h = h / repeats;

    dst[idx] = src[b * kv_heads * seq * head_dim + kv_h * seq * head_dim + s * head_dim + d];
}

void gqa_expand(const float* src, float* dst,
                int batch, int kv_heads, int n_heads, int seq, int head_dim) {
    int total = batch * n_heads * seq * head_dim;
    gqa_expand_kernel<<<(total + 255) / 256, 256>>>(src, dst, batch, kv_heads, n_heads, seq, head_dim);
}

/* ─── GQA Reduce: sum query head gradients back to KV heads ──── */
/* src: [batch, n_heads, seq, head_dim] → dst: [batch, kv_heads, seq, head_dim]
 * Sum groups of (n_heads / kv_heads) query heads into each KV head.
 */

__global__ void gqa_reduce_kernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int batch, int kv_heads, int n_heads, int seq, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * seq * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int s = (idx / head_dim) % seq;
    int kv_h = (idx / (head_dim * seq)) % kv_heads;
    int b = idx / (head_dim * seq * kv_heads);

    int repeats = n_heads / kv_heads;
    float sum = 0.0f;
    for (int r = 0; r < repeats; r++) {
        int h = kv_h * repeats + r;
        sum += src[b * n_heads * seq * head_dim + h * seq * head_dim + s * head_dim + d];
    }
    dst[idx] = sum;
}

void gqa_reduce(const float* src, float* dst,
                int batch, int kv_heads, int n_heads, int seq, int head_dim) {
    int total = batch * kv_heads * seq * head_dim;
    gqa_reduce_kernel<<<(total + 255) / 256, 256>>>(src, dst, batch, kv_heads, n_heads, seq, head_dim);
}

/* ─── Scale tensor: y = x * scale ────────────────────────────── */

__global__ void scale_kernel(const float* __restrict__ x, float* __restrict__ y, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * scale;
}

void scale_tensor(const float* x, float* y, float scale, int n) {
    scale_kernel<<<(n + 255) / 256, 256>>>(x, y, scale, n);
}

/* ─── Transpose: [batch, seq, heads, dim] → [batch, heads, seq, dim] ── */

__global__ void transpose_0213_kernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int batch, int dim1, int dim2, int dim3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim1 * dim2 * dim3;
    if (idx >= total) return;

    int d3 = idx % dim3;
    int d2 = (idx / dim3) % dim2;
    int d1 = (idx / (dim3 * dim2)) % dim1;
    int b  = idx / (dim3 * dim2 * dim1);

    // src[b, d1, d2, d3] → dst[b, d2, d1, d3]
    int dst_idx = b * dim2 * dim1 * dim3 + d2 * dim1 * dim3 + d1 * dim3 + d3;
    dst[dst_idx] = src[idx];
}

void transpose_0213(const float* src, float* dst,
                    int batch, int dim1, int dim2, int dim3) {
    int total = batch * dim1 * dim2 * dim3;
    transpose_0213_kernel<<<(total + 255) / 256, 256>>>(src, dst, batch, dim1, dim2, dim3);
}

} /* extern "C" */
