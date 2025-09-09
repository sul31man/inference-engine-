#pragma once
#include "infer_engine/core/tensor.hpp"
#include <tuple>

namespace ie {
namespace ops {

/**
 * Apply Rotary Position Embedding (RoPE) to query and key tensors
 * 
 * @param q Query tensor [..., H, D]
 * @param k Key tensor [..., H, D] 
 * @param pos Position indices or precomputed cos/sin values
 * @param rotary_dim Number of dimensions to apply rotation to (0 = all)
 * @param theta_base Base for computing rotation frequencies
 * @return Tuple of (rotated_q, rotated_k)
 */
std::tuple<TensorView, TensorView> rope_apply(
    const TensorView& q, 
    const TensorView& k, 
    const TensorView& pos,
    int rotary_dim = 0,
    float theta_base = 10000.0f
);

} // namespace ops
} // namespace ie
