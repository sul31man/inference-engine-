#pragma once
#include "infer_engine/core/tensor.hpp"

namespace ie {
namespace ops {

/**
 * Linear transformation: y = x @ W.T + bias
 * 
 * @param x Input tensor [N, D_in]
 * @param W Weight tensor [D_out, D_in] 
 * @param bias Optional bias tensor [D_out] (can be null)
 * @return Output tensor [N, D_out]
 */
Tensor linear(const TensorView& x, const TensorView& W, const TensorView* bias = nullptr);

} // namespace ops
} // namespace ie
