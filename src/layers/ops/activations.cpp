#include "infer_engine/layers/ops/activations.hpp"
#include "infer_engine/core/tensor.hpp"
#include <cmath>

namespace ie {
namespace ops {

Tensor silu(const TensorView& x) {
    // SiLU activation: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    // Apply element-wise to input tensor
    
    auto output = Tensor::empty(x.shape, x.dt);

    const float* input_ptr = x.ptr<const float>();
    float* output_ptr = output.view.ptr<float>(); 
     
    // Use the built-in numel() method instead of manual calculation
    int64_t total_elements = x.numel();

    for(int64_t i = 0; i < total_elements; i++){
        float sigmoid_val = 1.0f / (1.0f + std::expf(-input_ptr[i]));
        output_ptr[i] = input_ptr[i] * sigmoid_val;
    }

    return output;  // Return the Tensor, not the view
}

Tensor gelu(const TensorView& x, bool approximate) {
    // GELU activation
    // If approximate=true, use tanh approximation:
    // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    auto output = Tensor::empty(x.shape, x.dt);
    
    const float* input_ptr = x.ptr<const float>();
    float* output_ptr = output.view.ptr<float>();
    
    int64_t total_elements = x.numel();
    
    if (approximate) {
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        for(int64_t i = 0; i < total_elements; i++){
            float xi = input_ptr[i];
            float inner = sqrt_2_over_pi * (xi + 0.044715f * xi * xi * xi);
            output_ptr[i] = 0.5f * xi * (1.0f + std::tanh(inner));
        }
    } else {
        // Exact GELU using erf function
        for(int64_t i = 0; i < total_elements; i++){
            float xi = input_ptr[i];
            output_ptr[i] = 0.5f * xi * (1.0f + std::erf(xi / std::sqrt(2.0f)));
        }
    }
    
    return output;
}

} // namespace ops
} // namespace ie
