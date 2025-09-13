#pragma once
#include "infer_engine/core/tensor.hpp"
#include <cstdint>

namespace ie {

// Convert BF16 to F32
inline float bf16_to_f32(uint16_t bf16_val) {
    // BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
    // F32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits
    // BF16 is basically F32 with truncated mantissa
    union {
        uint32_t as_uint32;
        float as_float;
    } f32_val;
    
    // BF16 to F32: shift left by 16 bits (add 16 zero mantissa bits)
    f32_val.as_uint32 = static_cast<uint32_t>(bf16_val) << 16;
    return f32_val.as_float;
}

// Convert BF16 tensor to F32 tensor
Tensor convert_bf16_to_f32(const TensorView& bf16_tensor);

} // namespace ie
