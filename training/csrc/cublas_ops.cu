/*
 * cuBLAS matmul wrapper for native training.
 *
 * All linear layers (in_proj, out_proj, output projection) use cuBLAS sgemm.
 * cuBLAS operates in column-major order; our tensors are row-major.
 * We handle this by swapping A/B and transposing: C^T = B^T @ A^T.
 *
 * This wrapper provides a row-major interface on top of column-major cuBLAS.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

static cublasHandle_t g_cublas_handle = nullptr;

/* ─── BF16 conversion kernel (defined before extern "C" for __global__) ───── */

__global__ void f32_to_bf16_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __bfloat162float(in[idx]);
    }
}

extern "C" {

/* Initialize cuBLAS handle (call once at training start). */
void cublas_init() {
    if (g_cublas_handle == nullptr) {
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS init failed: %d\n", status);
        }
        /* Enable TF32 tensor core math for all sgemm calls.
         * A100+: 156 TFLOPS (TF32) vs 19.5 TFLOPS (f32 CUDA cores).
         * 10-bit mantissa is sufficient precision for training. */
        cublasSetMathMode(g_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
}

/* Destroy cuBLAS handle. */
void cublas_destroy() {
    if (g_cublas_handle != nullptr) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

/*
 * Row-major matrix multiply: C = A @ B
 *
 * A: [m, k] row-major
 * B: [k, n] row-major
 * C: [m, n] row-major
 *
 * cuBLAS is column-major, so we compute: C^T = B^T @ A^T
 * which gives us C in row-major layout.
 */
void matmul_f32(
    const float* A, const float* B, float* C,
    int m, int n, int k
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    /* cuBLAS column-major trick:
     * Row-major C[m,n] = A[m,k] @ B[k,n]
     * is equivalent to:
     * Col-major C^T[n,m] = B^T[n,k] @ A^T[k,m]
     * cublasSgemm(N, N, n, m, k, &alpha, B, n, A, k, &beta, C, n)
     */
    cublasSgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        B, n,    /* B treated as col-major [n,k] = B^T */
        A, k,    /* A treated as col-major [k,m] = A^T */
        &beta,
        C, n     /* C stored as col-major [n,m] = C^T = row-major [m,n] */
    );
}

/*
 * Row-major: C = A @ B^T
 *
 * A: [m, k] row-major
 * B: [n, k] row-major (transposed: B^T is [k, n])
 * C: [m, n] row-major
 */
void matmul_f32_bt(
    const float* A, const float* B, float* C,
    int m, int n, int k
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    /* C[m,n] = A[m,k] @ B^T[k,n] where B is [n,k]
     * Col-major: C^T[n,m] = B[n,k] @ A^T[k,m]
     * but B is row-major [n,k] = col-major B^T[k,n], so we need CUBLAS_OP_T on B
     * Actually: cublasSgemm(T, N, n, m, k, alpha, B, k, A, k, beta, C, n)
     */
    cublasSgemm(g_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        B, k,    /* B is row-major [n,k], col-major [k,n], transpose to get [n,k] */
        A, k,    /* A treated as col-major [k,m] = A^T */
        &beta,
        C, n
    );
}

/*
 * Row-major: C = A^T @ B
 *
 * A: [k, m] row-major (transposed: A^T is [m, k])
 * B: [k, n] row-major
 * C: [m, n] row-major
 */
void matmul_f32_at(
    const float* A, const float* B, float* C,
    int m, int n, int k
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, m, k,
        &alpha,
        B, n,
        A, m,    /* A is row-major [k,m], col-major [m,k], transpose to get [k,m] */
        &beta,
        C, n
    );
}

/*
 * Row-major: C += A^T @ B (accumulate, beta=1)
 * Used for weight gradient accumulation across micro-batches.
 */
void matmul_f32_at_accum(
    const float* A, const float* B, float* C,
    int m, int n, int k
) {
    float alpha = 1.0f;
    float beta = 1.0f;

    cublasSgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, m, k,
        &alpha,
        B, n,
        A, m,
        &beta,
        C, n
    );
}

/*
 * Row-major: C += A @ B (accumulate, beta=1)
 */
void matmul_f32_accum(
    const float* A, const float* B, float* C,
    int m, int n, int k
) {
    float alpha = 1.0f;
    float beta = 1.0f;

    cublasSgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        B, n,
        A, k,
        &beta,
        C, n
    );
}

/*
 * Batched strided matrix multiply: C[i] = A[i] @ B[i] for i in 0..batch_count
 *
 * A: [batch_count, m, k] (strideA = m*k)
 * B: [batch_count, k, n] (strideB = k*n)
 * C: [batch_count, m, n] (strideC = m*n)
 */
void matmul_f32_batched(
    const float* A, const float* B, float* C,
    int m, int n, int k, int batch_count
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    long long strideA = (long long)m * k;
    long long strideB = (long long)k * n;
    long long strideC = (long long)m * n;

    cublasSgemmStridedBatched(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        B, n, strideB,
        A, k, strideA,
        &beta,
        C, n, strideC,
        batch_count
    );
}

/*
 * Batched: C[i] = A[i] @ B[i]^T for i in 0..batch_count
 *
 * A: [batch_count, m, k] (strideA = m*k)
 * B: [batch_count, n, k] (strideB = n*k) — transposed in the multiply
 * C: [batch_count, m, n] (strideC = m*n)
 */
void matmul_f32_bt_batched(
    const float* A, const float* B, float* C,
    int m, int n, int k, int batch_count
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    long long strideA = (long long)m * k;
    long long strideB = (long long)n * k;
    long long strideC = (long long)m * n;

    cublasSgemmStridedBatched(g_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        B, k, strideB,
        A, k, strideA,
        &beta,
        C, n, strideC,
        batch_count
    );
}

/*
 * Batched: C[i] = A[i]^T @ B[i] for i in 0..batch_count
 *
 * A: [batch_count, k, m] (strideA = k*m)
 * B: [batch_count, k, n] (strideB = k*n)
 * C: [batch_count, m, n] (strideC = m*n)
 */
void matmul_f32_at_batched(
    const float* A, const float* B, float* C,
    int m, int n, int k, int batch_count
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    long long strideA = (long long)k * m;
    long long strideB = (long long)k * n;
    long long strideC = (long long)m * n;

    cublasSgemmStridedBatched(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, m, k,
        &alpha,
        B, n, strideB,
        A, m, strideA,
        &beta,
        C, n, strideC,
        batch_count
    );
}

/* ─── BF16 Mixed-Precision Support ─────────────────────────────────────────── */

/*
 * FP32 → BF16 conversion kernel.
 * Simple element-wise truncation (BF16 = upper 16 bits of FP32 with rounding).
 */
void convert_f32_to_bf16(const float* in, __nv_bfloat16* out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    f32_to_bf16_kernel<<<blocks, threads>>>(in, out, n);
}

void convert_bf16_to_f32(const __nv_bfloat16* in, float* out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bf16_to_f32_kernel<<<blocks, threads>>>(in, out, n);
}

/*
 * BF16 matmul wrappers: accept FP32 inputs + BF16 scratch, convert, then GemmEx.
 * Output is always FP32. Compute is FP32 (accumulate in full precision).
 * Uses BF16 tensor cores on A100+ (312 TFLOPS vs 156 TFLOPS for TF32).
 */

/* C = A @ B (row-major) */
void matmul_bf16_from_f32(
    const float* A, const float* B, float* C,
    __nv_bfloat16* scratch_a, __nv_bfloat16* scratch_b,
    int m, int n, int k
) {
    int threads = 256;
    int a_n = m * k, b_n = k * n;
    f32_to_bf16_kernel<<<(a_n+threads-1)/threads, threads>>>(A, scratch_a, a_n);
    f32_to_bf16_kernel<<<(b_n+threads-1)/threads, threads>>>(B, scratch_b, b_n);

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        scratch_b, CUDA_R_16BF, n,
        scratch_a, CUDA_R_16BF, k,
        &beta,
        C, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

/* C = A @ B^T (row-major) */
void matmul_bf16_from_f32_bt(
    const float* A, const float* B, float* C,
    __nv_bfloat16* scratch_a, __nv_bfloat16* scratch_b,
    int m, int n, int k
) {
    int threads = 256;
    int a_n = m * k, b_n = n * k;
    f32_to_bf16_kernel<<<(a_n+threads-1)/threads, threads>>>(A, scratch_a, a_n);
    f32_to_bf16_kernel<<<(b_n+threads-1)/threads, threads>>>(B, scratch_b, b_n);

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        scratch_b, CUDA_R_16BF, k,
        scratch_a, CUDA_R_16BF, k,
        &beta,
        C, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

/* C += A^T @ B (row-major, accumulate) */
void matmul_bf16_from_f32_at_accum(
    const float* A, const float* B, float* C,
    __nv_bfloat16* scratch_a, __nv_bfloat16* scratch_b,
    int m, int n, int k
) {
    int threads = 256;
    int a_n = k * m, b_n = k * n;
    f32_to_bf16_kernel<<<(a_n+threads-1)/threads, threads>>>(A, scratch_a, a_n);
    f32_to_bf16_kernel<<<(b_n+threads-1)/threads, threads>>>(B, scratch_b, b_n);

    float alpha = 1.0f, beta = 1.0f;
    cublasGemmEx(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, m, k,
        &alpha,
        scratch_b, CUDA_R_16BF, n,
        scratch_a, CUDA_R_16BF, m,
        &beta,
        C, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

/* Batched: C[i] = A[i] @ B[i] */
void matmul_bf16_from_f32_batched(
    const float* A, const float* B, float* C,
    __nv_bfloat16* scratch_a, __nv_bfloat16* scratch_b,
    int m, int n, int k, int batch_count
) {
    int threads = 256;
    int a_total = batch_count * m * k, b_total = batch_count * k * n;
    f32_to_bf16_kernel<<<(a_total+threads-1)/threads, threads>>>(A, scratch_a, a_total);
    f32_to_bf16_kernel<<<(b_total+threads-1)/threads, threads>>>(B, scratch_b, b_total);

    float alpha = 1.0f, beta = 0.0f;
    long long strideA = (long long)m * k;
    long long strideB = (long long)k * n;
    long long strideC = (long long)m * n;

    cublasGemmStridedBatchedEx(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        scratch_b, CUDA_R_16BF, n, strideB,
        scratch_a, CUDA_R_16BF, k, strideA,
        &beta,
        C, CUDA_R_32F, n, strideC,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

/* Batched: C[i] = A[i] @ B[i]^T */
void matmul_bf16_from_f32_bt_batched(
    const float* A, const float* B, float* C,
    __nv_bfloat16* scratch_a, __nv_bfloat16* scratch_b,
    int m, int n, int k, int batch_count
) {
    int threads = 256;
    int a_total = batch_count * m * k, b_total = batch_count * n * k;
    f32_to_bf16_kernel<<<(a_total+threads-1)/threads, threads>>>(A, scratch_a, a_total);
    f32_to_bf16_kernel<<<(b_total+threads-1)/threads, threads>>>(B, scratch_b, b_total);

    float alpha = 1.0f, beta = 0.0f;
    long long strideA = (long long)m * k;
    long long strideB = (long long)n * k;
    long long strideC = (long long)m * n;

    cublasGemmStridedBatchedEx(g_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        scratch_b, CUDA_R_16BF, k, strideB,
        scratch_a, CUDA_R_16BF, k, strideA,
        &beta,
        C, CUDA_R_32F, n, strideC,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

/* Batched: C[i] = A[i]^T @ B[i] */
void matmul_bf16_from_f32_at_batched(
    const float* A, const float* B, float* C,
    __nv_bfloat16* scratch_a, __nv_bfloat16* scratch_b,
    int m, int n, int k, int batch_count
) {
    int threads = 256;
    int a_total = batch_count * k * m, b_total = batch_count * k * n;
    f32_to_bf16_kernel<<<(a_total+threads-1)/threads, threads>>>(A, scratch_a, a_total);
    f32_to_bf16_kernel<<<(b_total+threads-1)/threads, threads>>>(B, scratch_b, b_total);

    float alpha = 1.0f, beta = 0.0f;
    long long strideA = (long long)k * m;
    long long strideB = (long long)k * n;
    long long strideC = (long long)m * n;

    cublasGemmStridedBatchedEx(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, m, k,
        &alpha,
        scratch_b, CUDA_R_16BF, n, strideB,
        scratch_a, CUDA_R_16BF, m, strideA,
        &beta,
        C, CUDA_R_32F, n, strideC,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

} /* extern "C" */
