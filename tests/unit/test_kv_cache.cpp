#include "infer_engine/runtime/kv_cache.hpp"
#include "infer_engine/core/tensor.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace ie;

    KVCacheConfig cfg;
    cfg.num_layers = 2;
    cfg.max_seq_len = 4;
    cfg.num_q_heads = 3;
    cfg.num_kv_heads = 3;
    cfg.head_dim = 8;
    cfg.dtype = DType::F16; // KV cache now FP16

    std::cout << "Testing KVCache implementation...\n";

    KVCache cache(cfg);

    // Test: Verify cache layout
    TensorView kv = cache.k_view();
    TensorView vv = cache.v_view();
    
    std::vector<int64_t> expected_shape = {cfg.num_layers, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim};
    assert(kv.shape == expected_shape);
    assert(vv.shape == expected_shape);
    std::cout << "✓ Cache layout correct: [" << cfg.num_layers << ", " << cfg.max_seq_len 
              << ", " << cfg.num_kv_heads << ", " << cfg.head_dim << "]\n";

    // Test: Append K,V for layer 0, pos 0
    Tensor K = Tensor::empty({cfg.num_kv_heads, cfg.head_dim}, cfg.dtype);
    Tensor V = Tensor::empty({cfg.num_kv_heads, cfg.head_dim}, cfg.dtype);

    // Fill with test values
    uint16_t* kp = K.view.ptr<uint16_t>();
    uint16_t* vp = V.view.ptr<uint16_t>();
    auto f32_to_f16 = [](float f) -> uint16_t {
        union { uint32_t u; float f; } in; in.f = f;
        uint32_t x = in.u; uint32_t sign = (x >> 31) & 0x1; int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15; uint32_t mant = x & 0x7FFFFF; uint16_t out;
        if (exp <= 0) { if (exp < -10) out = (uint16_t)(sign << 15); else { mant |= 0x800000; uint32_t t = mant >> (1 - exp + 13); if ((mant >> (1 - exp + 12)) & 1) t += 1; out = (uint16_t)((sign << 15) | (t & 0x3FF)); } }
        else if (exp >= 31) { out = (uint16_t)((sign << 15) | (0x1F << 10) | (mant ? 0x200 : 0)); }
        else { uint16_t e = (uint16_t)exp & 0x1F; uint16_t m = (uint16_t)(mant >> 13); if (mant & 0x1000) { m += 1; if (m == 0x400) { m = 0; e += 1; if (e >= 31) { out = (uint16_t)((sign << 15) | (0x1F << 10)); return out; } } } out = (uint16_t)((sign << 15) | (e << 10) | (m & 0x3FF)); }
        return out;
    };
    for (int i = 0; i < cfg.num_kv_heads * cfg.head_dim; ++i) {
        kp[i] = f32_to_f16(static_cast<float>(i + 100));
        vp[i] = f32_to_f16(static_cast<float>(i + 200));
    }

    cache.append(0, 0, K.view, V.view);
    std::cout << "✓ Appended K,V to layer 0, pos 0\n";

    // Test: Verify values were stored correctly
    const uint16_t* cache_k = kv.ptr<const uint16_t>();
    const uint16_t* cache_v = vv.ptr<const uint16_t>();
    auto f16_to_f32 = [](uint16_t h) -> float {
        uint32_t sign = (h & 0x8000) << 16; uint32_t exp = (h & 0x7C00) >> 10; uint32_t mant = (h & 0x03FF); uint32_t f;
        if (exp == 0) { if (mant == 0) { f = sign; } else { exp = 127 - 15 + 1; while ((mant & 0x0400) == 0) { mant <<= 1; exp--; } mant &= 0x03FF; f = sign | (exp << 23) | (mant << 13); } }
        else if (exp == 0x1F) { f = sign | 0x7F800000 | (mant << 13); }
        else { exp = exp - 15 + 127; f = sign | (exp << 23) | (mant << 13); }
        union { uint32_t u; float f; } out{f}; return out.f;
    };
    
    // Check values at position [layer=0, seq=0, :, :]
    for (int h = 0; h < cfg.num_kv_heads; ++h) {
        for (int d = 0; d < cfg.head_dim; ++d) {
            int flat_idx = 0 * (cfg.max_seq_len * cfg.num_kv_heads * cfg.head_dim) +  // layer 0
                          0 * (cfg.num_kv_heads * cfg.head_dim) +                      // seq 0  
                          h * cfg.head_dim + d;                                     // head h, dim d
            
            float expected_k = static_cast<float>(h * cfg.head_dim + d + 100);
            float expected_v = static_cast<float>(h * cfg.head_dim + d + 200);
            
            assert(f16_to_f32(cache_k[flat_idx]) == expected_k);
            assert(f16_to_f32(cache_v[flat_idx]) == expected_v);
        }
    }
    std::cout << "✓ Values stored and retrieved correctly\n";

    // Test: Append to different position
    cache.append(1, 2, K.view, V.view);  // layer 1, seq pos 2
    std::cout << "✓ Appended to layer 1, pos 2\n";

    std::cout << "All KVCache tests passed!\n";
    return 0;
}


