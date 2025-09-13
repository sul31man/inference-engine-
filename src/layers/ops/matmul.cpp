#include "infer_engine/layers/ops/matmul.hpp"
#include <stdexcept>

namespace ie {
namespace ops {

Tensor matmul(const TensorView& A, const TensorView& B, bool transpose_b) {
    // Get shapes
    auto A_shape = A.shape; 
    auto B_shape = B.shape; 

    int64_t x = A_shape[0];
    int64_t y = A_shape[1];
    int64_t y_2 = B_shape[0];
    int64_t z = B_shape[1];

    // Handle transpose_b case
    if (transpose_b) {
        y_2 = B_shape[1];
        z = B_shape[0];
    }

    if(y != y_2){
        throw std::logic_error("dimensions are not compatible for matrix multiplication");
    }

    // Create output tensor
    Tensor output = Tensor::empty({x, z}, A.dt);
    const float* A_ptr = A.ptr<const float>();
    const float* B_ptr = B.ptr<const float>();
    float* output_ptr = output.view.ptr<float>();

    // Perform matrix multiplication
    for (int64_t i = 0; i < x; i++){
        for (int64_t j = 0; j < z; j++){
            float sum = 0.0f; 

            for (int64_t k = 0; k < y; k++){
                float b_val;
                if (transpose_b) {
                    b_val = B_ptr[j * B_shape[1] + k];
                } else {
                    b_val = B_ptr[k * z + j];
                }
                sum += A_ptr[i * y + k] * b_val;
            }
            
            output_ptr[i * z + j] = sum;
        }
    }

    return output;
}

} // namespace ops
} // namespace ie


