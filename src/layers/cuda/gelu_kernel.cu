//this should be embarrassingly parallel 


#include <cuda_runtime.h> 
#include <math.h> 

__global__ void geluKernel(float* x, float* y, int n ){



    int tid = threadIdx.x;
    int bid = blockIdx.x; 
    int gid = tid + bid*blockDim.x;

    if (gid < n){

        float val = x[gid];

        val = 0.5*val*(1 + tanh(sqrt(2/M_PI)*(val + 0.044715*pow(val, 3))));
        
        y[gid] = val; 

    }
}