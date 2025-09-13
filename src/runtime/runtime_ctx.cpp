#include "infer_engine/runtime/runtime_ctx.hpp"
#include "infer_engine/layers/attention_forward.hpp"
#include "infer_engine/layers/mlp_forward.hpp"
#include "infer_engine/layers/ops/rmsnorm.hpp"
#include "infer_engine/layers/ops/linear.hpp"
#include "infer_engine/model/weights.hpp"
#include <stdexcept>
#include <iostream>
#include <cstdint>

namespace ie {

static void init_kv(std::unique_ptr<KVCache>& kv, const ModelCfg& cfg, int64_t max_seq_len) {
    KVCacheConfig kcfg;
    kcfg.num_layers = cfg.n_layers;
    kcfg.max_seq_len = (max_seq_len > 0 ? max_seq_len : 2048);
    kcfg.num_q_heads = cfg.n_heads;
    kcfg.num_kv_heads = cfg.n_kv_heads;
    kcfg.head_dim = cfg.d_model / cfg.n_heads;
    kcfg.dtype = DType::F16; // fp16 KV cache
    kv = std::make_unique<KVCache>(kcfg);
    std::cout << "[Runtime] KV configured: L=" << kcfg.num_layers
              << " S=" << kcfg.max_seq_len
              << " KV_H=" << kcfg.num_kv_heads
              << " D=" << kcfg.head_dim
              << " dtype=F16" << std::endl;
}

RuntimeCtx::RuntimeCtx(const ModelCfg& cfg, const ModelWeights& weights)
    : cfg_(cfg), weights_(weights) {
    init_kv(kv_, cfg_, 2048);
}

RuntimeCtx::RuntimeCtx(const ModelCfg& cfg, const ModelWeights& weights, int64_t max_seq_len)
    : cfg_(cfg), weights_(weights) {
    init_kv(kv_, cfg_, max_seq_len);
}

Tensor RuntimeCtx::forward_decode(int32_t token_id, int64_t pos) {
    // 1) Lookup token embedding -> x
    if (token_id < 0 || token_id >= cfg_.vocab_size) {
        throw std::out_of_range("Invalid token_id");
    }
    
    // Get embedding for token_id: [d_model]
    TensorView embed_weights = weights_.get_token_embeddings();
    Tensor x = Tensor::empty({cfg_.d_model}, DType::F32);
    
    // Copy embedding row for token_id with dtype awareness (F32/BF16)
    if (embed_weights.dt == DType::F32) {
        const float* embed_ptr = embed_weights.ptr<float>() + token_id * cfg_.d_model;
        std::memcpy(x.view.data, embed_ptr, static_cast<size_t>(cfg_.d_model) * sizeof(float));
    } else if (embed_weights.dt == DType::BF16) {
        auto bf16_to_f32 = [](uint16_t h) -> float {
            union { uint32_t u; float f; } out;
            out.u = static_cast<uint32_t>(h) << 16;
            return out.f;
        };
        const uint16_t* row = embed_weights.ptr<const uint16_t>() + token_id * cfg_.d_model;
        float* dst = x.view.ptr<float>();
        for (int64_t d = 0; d < cfg_.d_model; ++d) dst[d] = bf16_to_f32(row[d]);
    } else if (embed_weights.dt == DType::F16) {
        // Simple F16->F32 converter (rounding not exact, acceptable for now)
        auto f16_to_f32 = [](uint16_t h) -> float {
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp = (h & 0x7C00) >> 10;
            uint32_t mant = (h & 0x03FF);
            uint32_t f;
            if (exp == 0) {
                if (mant == 0) {
                    f = sign; // zero
                } else {
                    // subnormal
                    exp = 127 - 15 + 1;
                    while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
                    mant &= 0x03FF;
                    f = sign | (exp << 23) | (mant << 13);
                }
            } else if (exp == 0x1F) {
                f = sign | 0x7F800000 | (mant << 13); // inf/NaN
            } else {
                exp = exp - 15 + 127;
                f = sign | (exp << 23) | (mant << 13);
            }
            union { uint32_t u; float f; } out{f};
            return out.f;
        };
        const uint16_t* row = embed_weights.ptr<const uint16_t>() + token_id * cfg_.d_model;
        float* dst = x.view.ptr<float>();
        for (int64_t d = 0; d < cfg_.d_model; ++d) dst[d] = f16_to_f32(row[d]);
    } else {
        throw std::runtime_error("Unsupported embedding dtype");
    }
    
    // 2) For each layer: rms_norm + attn_forward + rms_norm + mlp_forward
    for (int64_t layer_idx = 0; layer_idx < cfg_.n_layers; ++layer_idx) {
        LayerWeightsCXX layer_weights = weights_.get_layer_weights(layer_idx);
        
        // Pre-attention RMS normalization
        Tensor x_norm = ie::ops::rmsnorm(x.view, *layer_weights.input_layernorm);
        
        // Attention forward pass  
        ie::layers::AttentionConfig attn_cfg{cfg_.d_model, cfg_.n_heads, cfg_.n_kv_heads, cfg_.d_model / cfg_.n_heads, cfg_.rope_theta, cfg_.rope_dim};
        Tensor attn_out = ie::layers::attn_forward(x_norm.view, reinterpret_cast<const ie::layers::AttentionWeights&>(layer_weights.attn), attn_cfg, *kv_, layer_idx, pos);
        
        // Residual connection: x = x + attn_out
        float* x_ptr = x.view.ptr<float>();
        const float* attn_ptr = attn_out.view.ptr<float>();
        for (int64_t i = 0; i < cfg_.d_model; ++i) {
            x_ptr[i] += attn_ptr[i];
        }
        
        // Pre-MLP RMS normalization
        Tensor x_norm2 = ie::ops::rmsnorm(x.view, *layer_weights.post_attention_layernorm);
        
        // MLP forward pass
        ie::layers::MLPConfig mlp_cfg{cfg_.d_model, /*d_ff*/ cfg_.d_model * 4, /*use_gelu*/ true};
        Tensor mlp_out = ie::layers::mlp_forward(x_norm2.view, reinterpret_cast<const ie::layers::MLPWeights&>(layer_weights.mlp), mlp_cfg);
        
        // Residual connection: x = x + mlp_out
        const float* mlp_ptr = mlp_out.view.ptr<float>();
        for (int64_t i = 0; i < cfg_.d_model; ++i) {
            x_ptr[i] += mlp_ptr[i];
        }
    }
    
    // 3) Final RMS normalization
    TensorView final_norm_weights = weights_.get_final_norm();
    Tensor x_final = ie::ops::rmsnorm(x.view, final_norm_weights);
    
    // 4) Final projection to logits using linear op (handles BF16 weights safely)
    TensorView lm_head_weights = weights_.get_lm_head();
    TensorView x_row = make_view(x_final.view.data, x_final.view.dt, {1, cfg_.d_model});
    Tensor logits_mat = ie::ops::linear(x_row, lm_head_weights, nullptr); // [1, vocab]
    Tensor logits = Tensor::empty({cfg_.vocab_size}, DType::F32);
    std::memcpy(logits.view.data, logits_mat.view.ptr<float>(), static_cast<size_t>(cfg_.vocab_size) * sizeof(float));
    
    // 5) Return logits [vocab_size]
    return logits;
}

} // namespace ie
