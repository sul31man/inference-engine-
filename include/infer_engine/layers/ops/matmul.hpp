#pragma once
#include "infer_engine/core/tensor.hpp"

namespace ie {
namespace ops {

/**
 * Matrix multiply: C = A @ B (optionally transpose B).
 * A: [M, K], B: [K, N] or B^T: [N, K]. Returns [M, N].
 */
Tensor matmul(const TensorView& A, const TensorView& B, bool transpose_b = false);

} // namespace ops
} // namespace ie


