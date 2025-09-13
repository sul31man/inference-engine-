#pragma once
#include "infer_engine/core/tensor.hpp"
#include <cstdint>
#include <vector>
#include <memory>

namespace ie {

struct AttentionWeightsCXX {
    TensorView Wq;
    TensorView Wk;
    TensorView Wv;
    TensorView Wo;
    TensorView* bq = nullptr;
    TensorView* bk = nullptr;
    TensorView* bv = nullptr;
    TensorView* bo = nullptr;
};

struct MLPWeightsCXX {
    TensorView W1;
    TensorView W2;
    TensorView W3;
    TensorView* b1 = nullptr;
    TensorView* b2 = nullptr;
    TensorView* b3 = nullptr;
};

struct LayerWeightsCXX {
    AttentionWeightsCXX attn;
    MLPWeightsCXX mlp;
    TensorView* input_layernorm = nullptr;      // RMS norm before attention
    TensorView* post_attention_layernorm = nullptr;  // RMS norm before MLP
};

class ModelWeights {
public:
    // Embedding and output head
    TensorView get_token_embeddings() const;   // [vocab_size, d_model]
    TensorView get_lm_head() const;            // [vocab_size, d_model]
    TensorView get_final_norm() const;         // [d_model] - final RMS norm

    // Per-layer access
    LayerWeightsCXX get_layer_weights(int64_t layer_idx) const;

    // Binding APIs (call these to populate the structure)
    void set_token_embeddings(const TensorView& emb) { token_embeddings_ = emb; }
    void set_lm_head(const TensorView& head) { lm_head_ = head; }
    void set_final_norm(const TensorView& norm) { final_norm_ = norm; }
    void set_num_layers(int64_t n) { layers_.assign(static_cast<size_t>(n), LayerWeightsCXX{}); }
    void set_layer_weights(int64_t layer_idx, const LayerWeightsCXX& lw) {
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layers_.size()) {
            throw std::out_of_range("set_layer_weights: layer_idx out of range");
        }
        layers_[static_cast<size_t>(layer_idx)] = lw;
    }
    int64_t num_layers() const { return static_cast<int64_t>(layers_.size()); }

    // Keep backing storage alive (e.g., safetensors mmaps)
    void set_owner(const std::shared_ptr<void>& owner) { owner_ = owner; }

private:
    TensorView token_embeddings_{};
    TensorView lm_head_{};
    TensorView final_norm_{};
    std::vector<LayerWeightsCXX> layers_{};
    std::shared_ptr<void> owner_{}; // holds reader/mmap lifetime
};

} // namespace ie


