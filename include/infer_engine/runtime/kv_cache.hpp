#pragma once
#include "infer_engine/core/tensor.hpp"
#include <vector>
#include <cstdint>

namespace ie {

struct KVCacheConfig {
    int64_t num_layers{0};
    int64_t max_seq_len{0};
    int64_t num_q_heads{0};       // Query heads
    int64_t num_kv_heads{0};      // Key/Value heads (for GQA)
    int64_t head_dim{0};
    DType dtype{DType::F16};      // Store KV in fp16 to reduce memory
};

class KVCache {
public:
    explicit KVCache(const KVCacheConfig& cfg);

    // Append K and V for a given layer and sequence position.
    // K,V are expected as TensorView with shapes: [num_kv_heads, head_dim]
    void append(int64_t layer_idx, int64_t seq_pos, const TensorView& K, const TensorView& V);

    // Accessors to underlying storage views for inspection/testing
    // Layout is [layers][seq][kv_heads][d_head] 
    TensorView k_view() const;
    TensorView v_view() const;

    const KVCacheConfig& config() const { return cfg_; }

private:
    KVCacheConfig cfg_{};
    Tensor k_store_{}; // owns memory
    Tensor v_store_{}; // owns memory
};

} // namespace ie


