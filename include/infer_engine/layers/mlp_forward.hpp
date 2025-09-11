#pragma once
#include "infer_engine/core/tensor.hpp"

namespace ie {
namespace layers {

struct MLPWeights {
    TensorView W1;    // Gate projection [d_model, d_ff]
    TensorView W2;    // Down projection [d_ff, d_model]  
    TensorView W3;    // Up projection [d_model, d_ff]
    
    // Optional biases
    TensorView* b1 = nullptr;
    TensorView* b2 = nullptr; 
    TensorView* b3 = nullptr;
};

struct MLPConfig {
    int64_t d_model{0};
    int64_t d_ff{0};        // Feed-forward dimension (usually 4 * d_model)
    bool use_gelu{true};    // true = GELU, false = SiLU
};

/**
 * Gated MLP forward pass (SwiGLU/GeGLU style).
 * 
 * Standard gated MLP computation:
 * gate = activation(linear(x, W1, b1))     // [1, d_ff] 
 * up   = linear(x, W3, b3)                 // [1, d_ff]
 * hidden = gate * up                       // Element-wise multiply
 * output = linear(hidden, W2, b2)          // [1, d_model]
 * 
 * @param x Input tensor [1, d_model] or [d_model]
 * @param weights MLP weight matrices
 * @param config MLP configuration
 * @return MLP output, same shape as input
 */
Tensor mlp_forward(
    const TensorView& x,
    const MLPWeights& weights,
    const MLPConfig& config
);

} // namespace layers  
} // namespace ie
