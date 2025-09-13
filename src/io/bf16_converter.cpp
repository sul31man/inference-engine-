#include "bf16_converter.hpp"
#include <cstring>

namespace ie {

Tensor convert_bf16_to_f32(const TensorView& bf16_tensor) {
    if (bf16_tensor.dt != DType::BF16) {
        throw std::runtime_error("Input tensor is not BF16");
    }
    
    // Create F32 tensor with same shape
    Tensor f32_tensor = Tensor::empty(bf16_tensor.shape, DType::F32);
    
    // Convert data
    const uint16_t* bf16_data = bf16_tensor.ptr<uint16_t>();
    float* f32_data = f32_tensor.view.ptr<float>();
    
    int64_t numel = bf16_tensor.numel();
    for (int64_t i = 0; i < numel; ++i) {
        f32_data[i] = bf16_to_f32(bf16_data[i]);
    }
    
    return f32_tensor;
}

} // namespace ie
