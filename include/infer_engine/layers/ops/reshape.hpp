#pragma once
#include "infer_engine/core/tensor.hpp"
#include <vector>

namespace ie {
namespace ops {

/**
 * Reshape a tensor to a new shape (row-major, no copy if possible).
 * Caller ensures total elements match.
 */
Tensor reshape(const Tensor& x, const std::vector<int64_t>& new_shape);

} // namespace ops
} // namespace ie


