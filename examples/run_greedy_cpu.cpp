#include "infer_engine/model/config.hpp"
#include "infer_engine/model/weights.hpp"
#include "infer_engine/runtime/runtime_ctx.hpp"
#include "safetensors/reader.h" // placeholder if you add a C++ reader; else load via Python
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    using namespace ie;
    // TODO: parse args: model.json, model.safetensors, prompt
    std::string cfg_path = "model.json";
    std::string st_path = "model.safetensors";
    std::string prompt = "Hello";

    try {
        // 1) load model.json
        ModelCfg cfg = load_cfg(cfg_path);

        // 2) load weights (bind from safetensors) – you may implement a C++ reader or pre-load via Python
        // Placeholder: assume weights already constructed
        ModelWeights weights{/* fill with views to tensors */};

        // 3) tokenize prompt – placeholder (user to implement)
        std::vector<int32_t> tokens; // = tokenize(prompt)

        // 4) build runtime
        RuntimeCtx rt(cfg, weights);

        // 5) prefill prompt
        for (int64_t i = 0; i < (int64_t)tokens.size(); ++i) {
            // forward decode at position i; discard logits for prefill
            // rt.forward_decode(tokens[i], i);
        }

        // 6) decode one token, print top-k logits
        int32_t last_token = tokens.empty() ? 0 : tokens.back();
        // Tensor logits = rt.forward_decode(last_token, (int64_t)tokens.size());
        // TODO: sort top-k and print token ids + scores
        std::cout << "run_greedy_cpu skeleton ready" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


