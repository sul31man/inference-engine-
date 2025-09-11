#include "infer_engine/layers/ops/elementwise.hpp"
#include "infer_engine/core/tensor.hpp"
#include <stdexcept>
#include <limits>

namespace ie {
namespace ops {

Tensor scale(const TensorView& x, float alpha) {
    // Create output tensor with same shape and dtype
    auto output = Tensor::empty(x.shape, x.dt);
    
    // Get typed pointers to data
    const float* input_ptr = x.ptr<const float>();
    float* output_ptr = output.view.ptr<float>();
    
    int64_t num_elements = x.numel();
    
    // Element-wise scaling: y = alpha * x
    for (int64_t i = 0; i < num_elements; ++i) {
        output_ptr[i] = alpha * input_ptr[i];
    }
    
    return output;
}

Tensor apply_causal_mask(const TensorView& scores, int64_t seq_pos) {
    // Create output tensor with same shape and dtype
    auto output = Tensor::empty(scores.shape, scores.dt);
    
    // Get typed pointers to data
    const float* input_ptr = scores.ptr<const float>();
    float* output_ptr = output.view.ptr<float>();
    
    int64_t num_elements = scores.numel();
    
    // Copy input to output first
    for (int64_t i = 0; i < num_elements; ++i) {
        output_ptr[i] = input_ptr[i];
    }
    
    // Apply causal mask: set positions > seq_pos to large negative value
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t i = seq_pos + 1; i < num_elements; ++i) {
        output_ptr[i] = neg_inf;
    }
    
    return output;
}

} // namespace ops
} // namespace ie

