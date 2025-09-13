#pragma once
#include "infer_engine/core/tensor.hpp"
#include "infer_engine/runtime/kv_cache.hpp"

namespace ie {
namespace layers {

struct AttentionWeights {
    TensorView Wq;    // [d_model, n_q_heads * head_dim]
    TensorView Wk;    // [d_model, n_kv_heads * head_dim] - GQA: fewer heads
    TensorView Wv;    // [d_model, n_kv_heads * head_dim] - GQA: fewer heads
    TensorView Wo;    // [n_q_heads * head_dim, d_model]
    
    // Optional biases
    TensorView* bq = nullptr;
    TensorView* bk = nullptr;
    TensorView* bv = nullptr;
    TensorView* bo = nullptr;
};

struct AttentionConfig {
    int64_t d_model{0};
    int64_t n_q_heads{0};         // Query heads (e.g., 32)
    int64_t n_kv_heads{0};        // Key/Value heads (e.g., 8 for GQA)
    int64_t head_dim{0};
    float rope_theta{10000.0f};
    int64_t rope_dim{0};  // 0 = use full head_dim
    
    // Computed properties
    int64_t gqa_group_size() const { return n_q_heads / n_kv_heads; }
};

/**
 * Single-layer attention forward pass for one token.
 * 
 * Steps:
 * 1. q,k,v projections from input
 * 2. Apply RoPE to q,k  
 * 3. Retrieve past K,V from cache + append current k,v
 * 4. Compute attention: scores = q @ K^T / sqrt(d), apply causal mask, softmax
 * 5. Context: ctx = softmax @ V
 * 6. Output projection: out = ctx @ Wo
 * 
 * @param x Input token [1, d_model] or [d_model]
 * @param weights Attention weight matrices
 * @param config Attention configuration  
 * @param cache KV cache to read from and append to
 * @param layer_idx Which transformer layer (for cache indexing)
 * @param seq_pos Current sequence position (for RoPE and cache)
 * @return Attention output [1, d_model] or [d_model]
 */
Tensor attn_forward(
    const TensorView& x,
    const AttentionWeights& weights,
    const AttentionConfig& config,
    KVCache& cache,
    int64_t layer_idx,
    int64_t seq_pos
);

} // namespace layers
} // namespace ie
