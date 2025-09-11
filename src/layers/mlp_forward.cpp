#include "infer_engine/layers/mlp_forward.hpp"
#include "infer_engine/layers/ops/linear.hpp"
#include "infer_engine/layers/ops/activations.hpp"
#include "infer_engine/layers/ops/elementwise.hpp"
#include <stdexcept>

namespace ie {
namespace layers {

Tensor mlp_forward(
    const TensorView& x,
    const MLPWeights& weights,
    const MLPConfig& config
) {
    // Step 1: Gate projection with activation
    //   gate_linear = linear(x, weights.W1, weights.b1)    // [1, d_ff]
    //   if (config.use_gelu):
    //       gate = gelu(gate_linear)
    //   else:
    //       gate = silu(gate_linear)
    //
    Tensor gate_linear = ie::ops::linear(x, weights.W1, weights.b1);
    Tensor gate;
    if(config.use_gelu){
        gate = ie::ops::gelu(gate_linear.view);
    }
    else{
        gate = ie::ops::silu(gate_linear.view);
    }
    
    // Step 2: Up projection (no activation)
    //   up = linear(x, weights.W3, weights.b3)             // [1, d_ff]
    //
    Tensor up = ie::ops::linear(x, weights.W3, weights.b3);
  
    // Step 3: Element-wise multiply (gating)
    //   hidden = gate * up                                 // [1, d_ff]
    //
    Tensor hidden = ie::ops::multiply(gate.view, up.view);
    
    // Step 4: Down projection
    //   output = linear(hidden, weights.W2, weights.b2)    // [1, d_model]
    //
    Tensor output = ie::ops::linear(hidden.view, weights.W2, weights.b2);
    
    return output;
}

} // namespace layers
} // namespace ie
