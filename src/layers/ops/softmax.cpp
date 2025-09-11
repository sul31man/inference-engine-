#include "infer_engine/layers/ops/softmax.hpp"
#include "infer_engine/core/tensor.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace ie {
namespace ops {

Tensor softmax(const TensorView& x, int axis) {
    // Handle negative axis (e.g., axis=-1 means last axis)
    if (axis < 0) {
        axis += x.rank();
    }
    
    // For now, implement simple case: 2D tensor, axis=-1 (last axis)
    // This does row-wise softmax
    if (x.rank() != 2 || axis != (x.rank() - 1)) {
        throw std::runtime_error("Softmax currently only supports 2D tensors with axis=-1");
    }
    
    // Create output tensor with same shape and dtype
    auto output = Tensor::empty(x.shape, x.dt);
    
    // Get typed pointers to data
    const float* input_ptr = x.ptr<const float>();
    float* output_ptr = output.view.ptr<float>();
    
    int64_t rows = x.shape[0];    // Number of rows
    int64_t cols = x.shape[1];    // Number of columns (softmax dimension)
    
    // Process each row independently
    for (int64_t row = 0; row < rows; ++row) {
        const float* row_input = input_ptr + row * cols;
        float* row_output = output_ptr + row * cols;
        
        // Step 1: Find maximum for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t col = 0; col < cols; ++col) {
            max_val = std::max(max_val, row_input[col]);
        }
        
        // Step 2: Compute sum of exp(x - max)
        float sum_exp = 0.0f;
        for (int64_t col = 0; col < cols; ++col) {
            float shifted = row_input[col] - max_val;
            sum_exp += std::expf(shifted);
        }
        
        // Step 3: Normalize
        for (int64_t col = 0; col < cols; ++col) {
            float shifted = row_input[col] - max_val;
            row_output[col] = std::expf(shifted) / sum_exp;
        }
    }
    
    return output;
}

} // namespace ops
} // namespace ie