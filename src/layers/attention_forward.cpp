#include "infer_engine/layers/attention_forward.hpp"
#include "infer_engine/layers/ops/linear.hpp"
#include "infer_engine/layers/ops/softmax.hpp"
#include "infer_engine/layers/ops/rope.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <cassert>
#ifdef IE_OMP
#include <omp.h>
#endif

namespace ie {
namespace layers {

Tensor attn_forward(
    const TensorView& x,
    const AttentionWeights& weights,
    const AttentionConfig& config,
    KVCache& cache,
    int64_t layer_idx,
    int64_t seq_pos
) {
    const int64_t n_q_heads = config.n_q_heads;
    const int64_t n_kv_heads = config.n_kv_heads;
    const int64_t d_head = config.head_dim;
    if (n_q_heads <= 0 || n_kv_heads <= 0 || d_head <= 0) {
        throw std::invalid_argument("Invalid attention config (n_q_heads/n_kv_heads/head_dim)");
    }
    if (n_q_heads % n_kv_heads != 0) {
        throw std::invalid_argument("n_q_heads must be divisible by n_kv_heads for GQA");
    }
    
    const int64_t gqa_group_size = n_q_heads / n_kv_heads;

    // Step 1: q, k, v projections
    // Q: x -> [1, n_q_heads*d_head], K,V: x -> [1, n_kv_heads*d_head]
    // Validate weight shapes early to avoid OOB
    const int64_t expected_q_out = n_q_heads * d_head;
    const int64_t expected_kv_out = n_kv_heads * d_head;
    if (weights.Wq.shape.size() != 2 || weights.Wq.shape[0] != expected_q_out || weights.Wq.shape[1] != x.shape.back()) {
        throw std::runtime_error("Wq shape mismatch");
    }
    if (weights.Wk.shape.size() != 2 || weights.Wk.shape[0] != expected_kv_out || weights.Wk.shape[1] != x.shape.back()) {
        throw std::runtime_error("Wk shape mismatch");
    }
    if (weights.Wv.shape.size() != 2 || weights.Wv.shape[0] != expected_kv_out || weights.Wv.shape[1] != x.shape.back()) {
        throw std::runtime_error("Wv shape mismatch");
    }
    Tensor q = ie::ops::linear(x, weights.Wq, weights.bq);
    Tensor k = ie::ops::linear(x, weights.Wk, weights.bk);
    Tensor v = ie::ops::linear(x, weights.Wv, weights.bv);

    // Create head views: Q has more heads than K,V in GQA
    TensorView q_heads = make_view(q.view.data, q.view.dt, {n_q_heads, d_head});
    TensorView k_heads = make_view(k.view.data, k.view.dt, {n_kv_heads, d_head});
    TensorView v_heads = make_view(v.view.data, v.view.dt, {n_kv_heads, d_head});

    // Step 2: RoPE on q, k (separate for Q and K since they have different head counts)
    const int64_t rotary_dim = (config.rope_dim > 0) ? config.rope_dim : d_head;
    const int64_t pairs = rotary_dim / 2;
    
    // RoPE for Q heads
    Tensor pos_q = Tensor::empty({n_q_heads, pairs, 2}, DType::F32);
    {
        float* pp = pos_q.view.ptr<float>();
        for (int64_t h = 0; h < n_q_heads; ++h) {
            for (int64_t i = 0; i < pairs; ++i) {
                float exponent = -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim);
                float theta_i = std::pow(config.rope_theta, exponent);
                float angle = static_cast<float>(seq_pos) * theta_i;
                int64_t base = h * (pairs * 2) + i * 2;
                pp[base + 0] = std::cos(angle);
                pp[base + 1] = std::sin(angle);
            }
        }
    }
    
    // RoPE for K heads  
    Tensor pos_k = Tensor::empty({n_kv_heads, pairs, 2}, DType::F32);
    {
        float* pp = pos_k.view.ptr<float>();
        for (int64_t h = 0; h < n_kv_heads; ++h) {
            for (int64_t i = 0; i < pairs; ++i) {
                float exponent = -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim);
                float theta_i = std::pow(config.rope_theta, exponent);
                float angle = static_cast<float>(seq_pos) * theta_i;
                int64_t base = h * (pairs * 2) + i * 2;
                pp[base + 0] = std::cos(angle);
                pp[base + 1] = std::sin(angle);
            }
        }
    }

    // Apply RoPE separately to Q and K
    auto [q_rot_t, q_unused_t] = ie::ops::rope_apply(q_heads, q_heads, pos_q.view, rotary_dim, config.rope_theta);
    auto [k_unused_t, k_rot_t] = ie::ops::rope_apply(k_heads, k_heads, pos_k.view, rotary_dim, config.rope_theta);
    TensorView q_rot_v = q_rot_t.view;
    TensorView k_rot_v = k_rot_t.view;

    // Step 3: Update KV cache with current K,V in FP16 and KVH layout
    // Sources here are already [n_kv_heads, d_head] in F32 (k_rot_v, v_heads)
    // Convert rowwise to fp16 and append
    auto f32_to_f16 = [](float f) -> uint16_t {
        union { uint32_t u; float f; } in; in.f = f;
        uint32_t x = in.u;
        uint32_t sign = (x >> 31) & 0x1;
        int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = x & 0x7FFFFF;
        uint16_t out;
        if (exp <= 0) {
            if (exp < -10) {
                out = (uint16_t)(sign << 15);
            } else {
                mant |= 0x800000;
                uint32_t t = mant >> (1 - exp + 13);
                if ((mant >> (1 - exp + 12)) & 1) t += 1;
                out = (uint16_t)((sign << 15) | (t & 0x3FF));
            }
        } else if (exp >= 31) {
            out = (uint16_t)((sign << 15) | (0x1F << 10) | (mant ? 0x200 : 0));
        } else {
            uint16_t e = (uint16_t)exp & 0x1F;
            uint16_t m = (uint16_t)(mant >> 13);
            if (mant & 0x1000) {
                m += 1;
                if (m == 0x400) { m = 0; e += 1; if (e >= 31) { out = (uint16_t)((sign << 15) | (0x1F << 10)); return out; } }
            }
            out = (uint16_t)((sign << 15) | (e << 10) | (m & 0x3FF));
        }
        return out;
    };

    assert(n_kv_heads > 0 && d_head > 0);
    Tensor Kf16 = Tensor::empty({n_kv_heads, d_head}, DType::F16);
    Tensor Vf16 = Tensor::empty({n_kv_heads, d_head}, DType::F16);
    const float* ksrc = k_rot_v.ptr<const float>();
    const float* vsrc = v_heads.ptr<const float>();
    uint16_t* kdst = Kf16.view.ptr<uint16_t>();
    uint16_t* vdst = Vf16.view.ptr<uint16_t>();
    for (int64_t h = 0; h < n_kv_heads; ++h) {
        const float* krow = ksrc + h * d_head;
        const float* vrow = vsrc + h * d_head;
        uint16_t* kdrow = kdst + h * d_head;
        uint16_t* vdrow = vdst + h * d_head;
        for (int64_t d = 0; d < d_head; ++d) {
            kdrow[d] = f32_to_f16(krow[d]);
            vdrow[d] = f32_to_f16(vrow[d]);
        }
    }
    cache.append(layer_idx, seq_pos, Kf16.view, Vf16.view);

    // Step 4: Compute attention scores with GQA - each K/V head serves multiple Q heads
    const int64_t seq_len = seq_pos + 1;
    Tensor scores = Tensor::empty({n_q_heads, seq_len}, DType::F32);
    float* scores_ptr = scores.view.ptr<float>();

    // Access cache memory for K,V (stored as F16)  
    TensorView Kcache = cache.k_view(); // [L, S, KV_H, D], F16
    TensorView Vcache = cache.v_view();
    const uint16_t* Kbase = Kcache.ptr<const uint16_t>();
    const uint16_t* Vbase = Vcache.ptr<const uint16_t>();

    const int64_t L = cache.config().num_layers;
    const int64_t S = cache.config().max_seq_len;
    const int64_t KV_H = cache.config().num_kv_heads;  // Number of KV heads (8 for Mistral)
    const int64_t D = cache.config().head_dim;

    const int64_t stride_L = S * KV_H * D; // elements per layer
    const int64_t stride_S = KV_H * D;     // elements per time step
    const int64_t stride_H = D;            // elements per kv head
    const int64_t layer_base = layer_idx * stride_L;

    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));

    const float* qptr = q_rot_v.ptr<const float>();
    auto f16_to_f32 = [](uint16_t h) -> float {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = (h & 0x7C00) >> 10;
        uint32_t mant = (h & 0x03FF);
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) { f = sign; }
            else {
                exp = 127 - 15 + 1;
                while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
                mant &= 0x03FF;
                f = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 0x1F) {
            f = sign | 0x7F800000 | (mant << 13);
        } else {
            exp = exp - 15 + 127;
            f = sign | (exp << 23) | (mant << 13);
        }
        union { uint32_t u; float f; } out{f};
        return out.f;
    };
    #ifdef IE_OMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t q_h = 0; q_h < n_q_heads; ++q_h) {
        // Map Q head to corresponding K/V head (GQA mapping)
        const int64_t kv_h = q_h / gqa_group_size;
        
        const float* qh = qptr + q_h * d_head;
        for (int64_t t = 0; t < seq_len; ++t) {
            const uint16_t* kvec = Kbase + layer_base + t * stride_S + kv_h * stride_H;
            float dot = 0.0f;
            for (int64_t d = 0; d < d_head; ++d) {
                dot += qh[d] * f16_to_f32(kvec[d]);
            }
            scores_ptr[q_h * seq_len + t] = dot * scale;
        }
    }

    // Step 5: softmax over time dim per head
    Tensor attn = ie::ops::softmax(scores.view, -1);
    const float* attn_ptr = attn.view.ptr<const float>();

    // Step 6: context = attn @ V (manual matmul with GQA mapping)
    Tensor ctx = Tensor::empty({n_q_heads, d_head}, DType::F32);
    float* ctx_ptr = ctx.view.ptr<float>();
    const uint16_t* Vb = Vcache.ptr<const uint16_t>();
    #ifdef IE_OMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t q_h = 0; q_h < n_q_heads; ++q_h) {
        // Map Q head to corresponding K/V head (same mapping as for attention scores)
        const int64_t kv_h = q_h / gqa_group_size;
        
        float* ch = ctx_ptr + q_h * d_head;
        for (int64_t d = 0; d < d_head; ++d) ch[d] = 0.0f;
        for (int64_t t = 0; t < seq_len; ++t) {
            float a = attn_ptr[q_h * seq_len + t];
            const uint16_t* vvec = Vb + layer_base + t * stride_S + kv_h * stride_H;
            for (int64_t d = 0; d < d_head; ++d) {
                ch[d] += a * f16_to_f32(vvec[d]);
            }
        }
    }

    // Step 7: output projection: flatten context and apply Wo
    TensorView ctx_flat = make_view(ctx.view.data, ctx.view.dt, {1, n_q_heads * d_head});
    Tensor out = ie::ops::linear(ctx_flat, weights.Wo, weights.bo);
    return out;
}

} // namespace layers
} // namespace ie
