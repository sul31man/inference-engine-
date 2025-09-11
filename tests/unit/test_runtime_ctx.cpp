#include "infer_engine/runtime/runtime_ctx.hpp"
#include "infer_engine/model/config.hpp"
#include "infer_engine/model/weights.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace ie;
    std::cout << "RuntimeCtx forward_decode test skeleton...\n";

    // Minimal cfg
    ModelCfg cfg{ /* d_model */ 64, /* n_layers */ 1, /* n_heads */ 4, /* vocab_size */ 32000, /* rope_theta */ 10000.0f, /* rope_dim */ 0 };

    // TODO: Construct ModelWeights with small tensors bound to views
    ModelWeights weights{ /* fill fields with TensorViews */ };

    RuntimeCtx rt(cfg, weights);

    // TODO: Call forward_decode and compare logits vs reference
    // Tensor logits = rt.forward_decode(42, /*pos=*/0);
    // assert(logits.view.shape == std::vector<int64_t>{cfg.vocab_size});

    std::cout << "RuntimeCtx test skeleton ready\n";
    return 0;
}


