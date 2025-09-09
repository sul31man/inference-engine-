#pragma once
#include "infer_engine/core/tensor.hpp"

namespace ie {
namespace ops {

/**
 * SiLU (Swish) activation: x * sigmoid(x)
 * 
 * @param x Input tensor
 * @return Output tensor, same shape as x
 */
TensorView silu(const TensorView& x);

/**
 * GELU activation with tanh approximation
 * 
 * @param x Input tensor
 * @param approximate Use tanh approximation if true
 * @return Output tensor, same shape as x
 */
TensorView gelu(const TensorView& x, bool approximate = true);

} // namespace ops
} // namespace ie
