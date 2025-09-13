#include <cudaruntime.h> 
#include "infer/softmax.cpp"
#include "inference_engine/linear.cpp"

//lets build the attention kernel. this will not be an implementation of Flash attention as this should be a basic bare minimum to get things running
//this will require us to project the tokens into q,k,v, mat mull between q and k, softmax it then mat mul with v. 


__global__ void attentionKernel(float* q, float* k, float* v, int rows, int mid, int cols){
 


    int tid = threadIdx; 
    int bid = blockIdx; 
    int gid = tid + bid*blockDim; 


    

}