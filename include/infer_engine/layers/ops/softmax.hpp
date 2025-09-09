#pragma once
#include "infer_engine/core/tensor.hpp"
#include <cmath>

namespace ie {
namespace ops {

/**
 * Stable softmax along specified axis
 * 
 * @param x Input tensor
 * @param axis Axis to apply softmax along (-1 for last axis)
 * @return Output tensor, same shape as x
 */
TensorView softmax(const TensorView& x, int axis = -1){

    //lets implement the softmax here

    float denom = 0; 
    int length = x.shape()[0];  // Get the size of the first dimension
    
    auto output = Tensor::empty(x.shape(), x.dt);

    

    
    for(int i = 0; i < length; i++){

        denom += std::expf(x.data()[i]);  // Use data() to access elements
    }

    for (int i = 0; i < length; i++){
        
        output.data()[i] = std::expf(x.data()[i]) / denom;  // Write to output, not input
    }

    return output;  // Return the output tensor
};



} // namespace ops
} // namespace ie
