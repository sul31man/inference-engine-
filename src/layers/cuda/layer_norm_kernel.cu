// Row-wise LayerNorm CUDA kernels
#include <cuda_runtime.h>

// Compute LayerNorm per row: for each row r in [0, rows), normalize
// x[r, :] with mean/var over the last dimension (cols), then apply
// y[r, c] = (x[r, c] - mean) / sqrt(var + eps) * gamma[c] + beta[c]

extern "C" __global__ void layernorm_rowwise(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int rows,
    int cols,
    float eps
) {
    extern __shared__ float smem[]; // size >= blockDim.x

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;

    // 1) mean
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        sum += xr[c];
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)cols;

    // 2) variance
    float vsum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float d = xr[c] - mean;
        vsum += d * d;
    }
    smem[tid] = vsum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float var = smem[0] / (float)cols;
    float inv_std = rsqrtf(var + eps);

    // 3) normalize and affine
    for (int c = tid; c < cols; c += blockDim.x) {
        float n = (xr[c] - mean) * inv_std;
        float g = gamma ? gamma[c] : 1.0f;
        float b = beta ? beta[c] : 0.0f;
        yr[c] = n * g + b;
    }
}

// Optional scalar-parameter variant (broadcast gamma/beta scalars)
extern "C" __global__ void layernorm_rowwise_scalar(
    const float* __restrict__ x,
    float* __restrict__ y,
    float gamma,
    float beta,
    int rows,
    int cols,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;

    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) sum += xr[c];
    smem[tid] = sum; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s) smem[tid] += smem[tid + s]; __syncthreads(); }
    float mean = smem[0] / (float)cols;

    float vsum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) { float d = xr[c] - mean; vsum += d*d; }
    smem[tid] = vsum; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s) smem[tid] += smem[tid + s]; __syncthreads(); }
    float var = smem[0] / (float)cols;
    float inv_std = rsqrtf(var + eps);

    for (int c = tid; c < cols; c += blockDim.x) {
        float n = (xr[c] - mean) * inv_std;
        yr[c] = n * gamma + beta;
    }
}