#pragma once
#include <cstdint>
#include <string>

namespace ie {

struct ModelCfg {
    int64_t d_model{0};
    int64_t n_layers{0};
    int64_t n_heads{0};           // Query heads (e.g., 32 for Mistral)
    int64_t n_kv_heads{0};        // Key/Value heads (e.g., 8 for Mistral GQA)
    int64_t vocab_size{0};
    float rope_theta{10000.0f};
    int64_t rope_dim{0};
    
    // Computed property
    int64_t head_dim() const { return d_model / n_heads; }
    int64_t kv_head_dim() const { return d_model / n_heads; } // Same as Q head dim
    int64_t gqa_group_size() const { return n_heads / n_kv_heads; } // How many Q heads per K/V head
};

// Load configuration from a JSON file (skeleton; implement parsing)
ModelCfg load_cfg(const std::string& path);

} // namespace ie


