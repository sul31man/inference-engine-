#include <cuda_runtime.h>


__device__ float warp_reduce_sum(float val) {
    // Warp-level reduction using shuffle primitives
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__global__ void softmaxKernel(float* logits, int n, float* softmax) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gid = tid + blockDim.x * bid;
    int wid = tid % 32;
    int lane = tid / 32;
    int numWarps = (blockDim.x + 31) / 32;

    if (gid < n) {
        __shared__ float tile[32]; // Max warps per block is typically 32

        float val = logits[gid];

        // Find max value first for numerical stability
        float maxVal = val;
        for (int offset = 16; offset > 0; offset >>= 1) {
            maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, offset));
        }
        maxVal = __shfl_sync(0xffffffff, maxVal, 0); // Broadcast to all threads in warp

        // Store warp max in shared memory
        if (wid == 0 && lane < numWarps) {
            tile[lane] = maxVal;
        }
        __syncthreads();

        // Find global max across all warps
        float globalMax = 0.0f;
        if (tid < numWarps) {
            globalMax = tile[tid];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            if (tid < numWarps) {
                globalMax = fmaxf(globalMax, __shfl_down_sync(0xffffffff, globalMax, offset));
            }
        }
        globalMax = __shfl_sync(0xffffffff, globalMax, 0);

        // Compute exp(val - globalMax)
        val = expf(val - globalMax);

        // Compute sum of exponentials
        float warpSum = warp_reduce_sum(val);

        if (wid == 0 && lane < numWarps) {
            tile[lane] = warpSum;
        }
        __syncthreads();

        float sum = 0.0f;
        for (int i = 0; i < numWarps; i++) {
            sum += tile[i];
        }

        // Normalize
        val = val / sum;
        softmax[gid] = val;
    }
}