#pragma once
#include "infer_engine/core/tensor.hpp"

namespace ie {
namespace ops {

/** Scale tensor by scalar factor (y = x * alpha). */
Tensor scale(const TensorView& x, float alpha);

/** Apply causal mask to attention scores up to seq_pos (inclusive). */
Tensor apply_causal_mask(const TensorView& scores, int64_t seq_pos);

} // namespace ops
} // namespace ie


