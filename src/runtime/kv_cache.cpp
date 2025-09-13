#include "infer_engine/runtime/kv_cache.hpp"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cassert>
namespace ie {

KVCache::KVCache(const KVCacheConfig& cfg) : cfg_(cfg) {
    // Allocate k_store_ and v_store_ with shape:
    // [num_layers, max_seq_len, num_kv_heads, head_dim] for GQA support
    std::vector<int64_t> shape{cfg_.num_layers, cfg_.max_seq_len, cfg_.num_kv_heads, cfg_.head_dim};
    k_store_ = Tensor::empty(shape, cfg_.dtype);
    v_store_ = Tensor::empty(shape, cfg_.dtype);

    // Log expected memory usage
    const double elem_bytes = static_cast<double>(dtype_bytes(cfg_.dtype));
    const double total_bytes = 2.0 * static_cast<double>(cfg_.num_layers)
        * static_cast<double>(cfg_.max_seq_len)
        * static_cast<double>(cfg_.num_kv_heads)
        * static_cast<double>(cfg_.head_dim) * elem_bytes;
    const double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(2)
              << "[KVCache] layers=" << cfg_.num_layers
              << " seq=" << cfg_.max_seq_len
              << " kv_heads=" << cfg_.num_kv_heads
              << " head_dim=" << cfg_.head_dim
              << " dtype=" << (cfg_.dtype == DType::F16 ? "F16" : "F32")
              << " -> expected ~" << total_gb << " GB" << std::endl;
}

static inline size_t kv_offset(int64_t L, int64_t S, int64_t KVH, int64_t D,
                               int64_t l, int64_t pos, int64_t kvh, int64_t d) {
    return static_cast<size_t>(((((uint64_t)l) * S + pos) * KVH + kvh) * (uint64_t)D + d);
}

void KVCache::append(int64_t layer_idx, int64_t seq_pos, const TensorView& K, const TensorView& V) {
    // Bounds check
    if (layer_idx < 0 || layer_idx >= cfg_.num_layers) {
        throw std::out_of_range("layer_idx out of bounds");
    }
    if (seq_pos < 0 || seq_pos >= cfg_.max_seq_len) {
        throw std::out_of_range("seq_pos out of bounds");
    }
    
    // Shape check - K,V should have kv_heads, not q_heads
    if (K.shape.size() != 2 || K.shape[0] != cfg_.num_kv_heads || K.shape[1] != cfg_.head_dim) {
        throw std::invalid_argument("K shape mismatch");
    }
    if (V.shape.size() != 2 || V.shape[0] != cfg_.num_kv_heads || V.shape[1] != cfg_.head_dim) {
        throw std::invalid_argument("V shape mismatch");
    }

    const int64_t KVH = cfg_.num_kv_heads;
    const int64_t D   = cfg_.head_dim;
    const int64_t S   = cfg_.max_seq_len;

    assert(seq_pos >= 0 && seq_pos < S);

    auto* kd = k_store_.view.ptr<uint8_t>();
    auto* vd = v_store_.view.ptr<uint8_t>();
    const auto* ks = K.ptr<const uint8_t>();
    const auto* vs = V.ptr<const uint8_t>();
    const size_t elem_b = dtype_bytes(cfg_.dtype); // 2 for F16
    const size_t row_bytes = static_cast<size_t>(D) * elem_b;

    for (int64_t kvh = 0; kvh < KVH; ++kvh) {
        assert(kvh >= 0 && kvh < KVH);
        const size_t dst_index = kv_offset(cfg_.num_layers, S, KVH, D, layer_idx, seq_pos, kvh, 0);
        uint8_t* kdst = kd + dst_index * elem_b;
        uint8_t* vdst = vd + dst_index * elem_b;

        const uint8_t* ksrc = ks + static_cast<size_t>(kvh) * row_bytes;
        const uint8_t* vsrc = vs + static_cast<size_t>(kvh) * row_bytes;

        // Debug log for per-step KV write
        std::cout << "[KVWrite] l=" << layer_idx << " pos=" << seq_pos
                  << " kvh=" << kvh << " bytes=" << row_bytes << std::endl;

        std::memcpy(kdst, ksrc, row_bytes);
        std::memcpy(vdst, vsrc, row_bytes);
    }
}

TensorView KVCache::k_view() const {
    return k_store_.view;
}

TensorView KVCache::v_view() const {
    return v_store_.view;
}

} // namespace ie


