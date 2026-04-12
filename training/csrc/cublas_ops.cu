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
#include <cstdio>

static cublasHandle_t g_cublas_handle = nullptr;

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

} /* extern "C" */
