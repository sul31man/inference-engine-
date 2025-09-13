#include "infer_engine/model/config.hpp"
#include "infer_engine/model/weights.hpp"
#include "infer_engine/runtime/runtime_ctx.hpp"
#include "infer_engine/core/tensor.hpp"
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <random>

int main(int argc, char** argv) {
    using namespace ie;
    // Prompt arg (byte-level toy)
    std::string prompt = (argc > 1) ? std::string(argv[1]) : std::string("hello");

    try {
        // Minimal demo config (toy model). Replace with load_cfg when implemented
        ModelCfg cfg;
        cfg.d_model = 32; cfg.n_layers = 2; cfg.n_heads = 4; cfg.vocab_size = 256; cfg.rope_theta = 10000.0f; cfg.rope_dim = 0;

        // 2) Build random toy weights and bind
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
        auto make_tensor = [&](const std::vector<int64_t>& shape){
            Tensor t = Tensor::empty(shape, ie::DType::F32);
            float* p = t.view.ptr<float>();
            for (int64_t i = 0; i < t.view.numel(); ++i) p[i] = dist(rng);
            return t;
        };

        ModelWeights weights;
        Tensor tok_emb = make_tensor({cfg.vocab_size, cfg.d_model});
        Tensor lm_head = make_tensor({cfg.vocab_size, cfg.d_model});
        weights.set_token_embeddings(tok_emb.view);
        weights.set_lm_head(lm_head.view);
        weights.set_num_layers(cfg.n_layers);
        const int64_t d_head = cfg.d_model / cfg.n_heads; const int64_t d_ff = 4 * cfg.d_model;
        for (int l = 0; l < cfg.n_layers; ++l){
            LayerWeightsCXX lw;
            Tensor Wq = make_tensor({cfg.d_model, cfg.n_heads * d_head});
            Tensor Wk = make_tensor({cfg.d_model, cfg.n_heads * d_head});
            Tensor Wv = make_tensor({cfg.d_model, cfg.n_heads * d_head});
            Tensor Wo = make_tensor({cfg.n_heads * d_head, cfg.d_model});
            lw.attn.Wq = Wq.view; lw.attn.Wk = Wk.view; lw.attn.Wv = Wv.view; lw.attn.Wo = Wo.view;
            Tensor W1 = make_tensor({cfg.d_model, d_ff});
            Tensor W3 = make_tensor({cfg.d_model, d_ff});
            Tensor W2 = make_tensor({d_ff, cfg.d_model});
            lw.mlp.W1 = W1.view; lw.mlp.W3 = W3.view; lw.mlp.W2 = W2.view;
            weights.set_layer_weights(l, lw);
        }

        // 3) tokenize prompt (byte-level)
        std::vector<int32_t> tokens; for (unsigned char c: prompt) tokens.push_back((int)c);

        // 4) build runtime
        RuntimeCtx rt(cfg, weights);

        // 5) prefill prompt
        for (int64_t i = 0; i < (int64_t)tokens.size(); ++i) {
            // forward decode at position i; discard logits for prefill
            // rt.forward_decode(tokens[i], i);
        }

        // 6) decode one token, print top-5 logits
        int32_t last_token = tokens.empty() ? 0 : tokens.back();
        Tensor logits = rt.forward_decode(last_token, (int64_t)tokens.size());
        struct P{int idx; float v;}; std::vector<P> v; const float* lp = logits.view.ptr<float>(); v.reserve(cfg.vocab_size);
        for (int i=0;i<cfg.vocab_size;++i) v.push_back({i, lp[i]});
        std::partial_sort(v.begin(), v.begin()+5, v.end(), [](const P&a,const P&b){return a.v>b.v;});
        std::cout << "Top-5 logits:\n"; for (int i=0;i<5;++i) std::cout << "  "<<v[i].idx<<": "<<v[i].v<<"\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


