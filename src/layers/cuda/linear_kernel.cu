#include <stdlib.h> 
#include <cudaruntime.h> 


__global__ void linearKernel(float* A, float* B, float* C ,int rows, int cols, int mid){


    int globalIdx.x = threadIdx.x + blockIdx.x*blockDim.x; 

    int globalIdx.y = threadIdx.y + blockIdx.y*blockDim.y;

    if(globalIdx.x < cols & globalIdx.y < rows){
       
        //lets create a simple implementation where we don't have to 

        float value = 0.0f; 

        for(int i=0; i < mid; i++){

            value += A[globalIdx.y*mid + i] * B[cols*i + globalIdx.x];


        }

        C[globalIdx.y*cols + globalIdx.x] = value; 

    }

}