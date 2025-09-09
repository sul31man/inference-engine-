#pragma once
#include "infer_engine/core/tensor.hpp"

namespace ie {
namespace ops {

/**
 * RMSNorm operation in fp32.
 * 
 * @param x Input tensor [..., D]
 * @param gamma Scale parameters [D]
 * @param eps Small constant for numerical stability
 * @return Normalized tensor, same shape as x
 */
TensorView rmsnorm(const TensorView& x, const TensorView& gamma, float eps = 1e-5f);

} // namespace ops
} // namespace ie
