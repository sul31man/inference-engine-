#include "infer_engine/runtime/kv_cache.hpp"
#include <stdexcept>
#include <cstring> 
namespace ie {

KVCache::KVCache(const KVCacheConfig& cfg) : cfg_(cfg) {
    // TODO: allocate k_store_ and v_store_ with shape:
    // [num_layers, max_seq_len, num_heads, head_dim]
    // Do not implement here per user request.

    std::vector<int64_t> shape{cfg_.num_layers, cfg_.max_seq_len, cfg_.num_heads, cfg_.head_dim};
    k_store_ = Tensor::empty(shape, cfg_.dtype);
    v_store_ = Tensor::empty(shape, cfg_.dtype);


}

void KVCache::append(int64_t layer_idx, int64_t seq_pos, const TensorView& K, const TensorView& V) {
    // Bounds check
    if (layer_idx < 0 || layer_idx >= cfg_.num_layers) {
        throw std::out_of_range("layer_idx out of bounds");
    }
    if (seq_pos < 0 || seq_pos >= cfg_.max_seq_len) {
        throw std::out_of_range("seq_pos out of bounds");
    }
    
    // Shape check
    if (K.shape.size() != 2 || K.shape[0] != cfg_.num_heads || K.shape[1] != cfg_.head_dim) {
        throw std::invalid_argument("K shape mismatch");
    }
    if (V.shape.size() != 2 || V.shape[0] != cfg_.num_heads || V.shape[1] != cfg_.head_dim) {
        throw std::invalid_argument("V shape mismatch");
    }

    const int64_t H = cfg_.num_heads; 
    const int64_t D = cfg_.head_dim; 

    const int64_t stride_L = cfg_.max_seq_len * H * D; 
    const int64_t stride_S = H * D; 

    const int64_t base = layer_idx * stride_L + seq_pos * stride_S; 
    const size_t bytes = static_cast<size_t>(H) * static_cast<size_t>(D) * dtype_bytes(cfg_.dtype);

    if (cfg_.dtype == DType::F32) {
        float* kd = k_store_.view.ptr<float>();
        float* vd = v_store_.view.ptr<float>();
        const float* ks = K.ptr<const float>();
        const float* vs = V.ptr<const float>();
        std::memcpy(kd + base, ks, bytes);
        std::memcpy(vd + base, vs, bytes);
    } else {
        // Generic byte copy for other dtypes
        auto* kd = k_store_.view.ptr<uint8_t>();
        auto* vd = v_store_.view.ptr<uint8_t>();
        const auto* ks = K.ptr<const uint8_t>();
        const auto* vs = V.ptr<const uint8_t>();
        std::memcpy(kd + base * dtype_bytes(cfg_.dtype), ks, bytes);
        std::memcpy(vd + base * dtype_bytes(cfg_.dtype), vs, bytes);
    } 
}

TensorView KVCache::k_view() const {
    return k_store_.view;
}

TensorView KVCache::v_view() const {
    return v_store_.view;
}

} // namespace ie


