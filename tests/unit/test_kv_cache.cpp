#include "infer_engine/runtime/kv_cache.hpp"
#include "infer_engine/core/tensor.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace ie;

    KVCacheConfig cfg;
    cfg.num_layers = 2;
    cfg.max_seq_len = 4;
    cfg.num_heads = 3;
    cfg.head_dim = 8;
    cfg.dtype = DType::F32;

    std::cout << "Testing KVCache implementation...\n";

    KVCache cache(cfg);

    // Test: Verify cache layout
    TensorView kv = cache.k_view();
    TensorView vv = cache.v_view();
    
    std::vector<int64_t> expected_shape = {cfg.num_layers, cfg.max_seq_len, cfg.num_heads, cfg.head_dim};
    assert(kv.shape == expected_shape);
    assert(vv.shape == expected_shape);
    std::cout << "✓ Cache layout correct: [" << cfg.num_layers << ", " << cfg.max_seq_len 
              << ", " << cfg.num_heads << ", " << cfg.head_dim << "]\n";

    // Test: Append K,V for layer 0, pos 0
    Tensor K = Tensor::empty({cfg.num_heads, cfg.head_dim}, cfg.dtype);
    Tensor V = Tensor::empty({cfg.num_heads, cfg.head_dim}, cfg.dtype);

    // Fill with test values
    float* kp = K.view.ptr<float>();
    float* vp = V.view.ptr<float>();
    for (int i = 0; i < cfg.num_heads * cfg.head_dim; ++i) {
        kp[i] = static_cast<float>(i + 100);  // K values: 100, 101, 102...
        vp[i] = static_cast<float>(i + 200);  // V values: 200, 201, 202...
    }

    cache.append(0, 0, K.view, V.view);
    std::cout << "✓ Appended K,V to layer 0, pos 0\n";

    // Test: Verify values were stored correctly
    const float* cache_k = kv.ptr<const float>();
    const float* cache_v = vv.ptr<const float>();
    
    // Check values at position [layer=0, seq=0, :, :]
    for (int h = 0; h < cfg.num_heads; ++h) {
        for (int d = 0; d < cfg.head_dim; ++d) {
            int flat_idx = 0 * (cfg.max_seq_len * cfg.num_heads * cfg.head_dim) +  // layer 0
                          0 * (cfg.num_heads * cfg.head_dim) +                      // seq 0  
                          h * cfg.head_dim + d;                                     // head h, dim d
            
            float expected_k = static_cast<float>(h * cfg.head_dim + d + 100);
            float expected_v = static_cast<float>(h * cfg.head_dim + d + 200);
            
            assert(cache_k[flat_idx] == expected_k);
            assert(cache_v[flat_idx] == expected_v);
        }
    }
    std::cout << "✓ Values stored and retrieved correctly\n";

    // Test: Append to different position
    cache.append(1, 2, K.view, V.view);  // layer 1, seq pos 2
    std::cout << "✓ Appended to layer 1, pos 2\n";

    std::cout << "All KVCache tests passed!\n";
    return 0;
}


