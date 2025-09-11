#pragma once
#include "infer_engine/core/tensor.hpp"

namespace ie {
namespace ops {

/**
 * Stable softmax along specified axis
 * 
 * @param x Input tensor
 * @param axis Axis to apply softmax along (-1 for last axis)
 * @return Output tensor, same shape as x
 */
Tensor softmax(const TensorView& x, int axis = -1);

} // namespace ops
} // namespace ie