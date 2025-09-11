#include "infer_engine/layers/attention_forward.hpp"
#include "infer_engine/layers/ops/linear.hpp"
#include "infer_engine/layers/ops/softmax.hpp"
#include "infer_engine/layers/ops/rope.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>

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
    const int64_t n_heads = config.n_heads;
    const int64_t d_head = config.head_dim;
    if (n_heads <= 0 || d_head <= 0) {
        throw std::invalid_argument("Invalid attention config (n_heads/head_dim)");
    }

    // Step 1: q, k, v projections: x -> [1, n_heads*d_head]
    Tensor q = ie::ops::linear(x, weights.Wq, weights.bq);
    Tensor k = ie::ops::linear(x, weights.Wk, weights.bk);
    Tensor v = ie::ops::linear(x, weights.Wv, weights.bv);

    // Create head views [n_heads, d_head]
    TensorView q_heads = make_view(q.view.data, q.view.dt, {n_heads, d_head});
    TensorView k_heads = make_view(k.view.data, k.view.dt, {n_heads, d_head});
    TensorView v_heads = make_view(v.view.data, v.view.dt, {n_heads, d_head});

    // Step 2: RoPE on q, k (build per-head cos/sin for this position)
    const int64_t rotary_dim = (config.rope_dim > 0) ? config.rope_dim : d_head;
    const int64_t pairs = rotary_dim / 2;
    Tensor pos = Tensor::empty({n_heads, pairs, 2}, DType::F32);
    {
        float* pp = pos.view.ptr<float>();
        for (int64_t h = 0; h < n_heads; ++h) {
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

    auto [q_rot_v, k_rot_v] = ie::ops::rope_apply(q_heads, k_heads, pos.view, rotary_dim, config.rope_theta);

    // Step 3: Update KV cache with current K,V
    cache.append(layer_idx, seq_pos, k_rot_v, v_heads);

    // Step 4: Compute attention scores against past K (including current), per head
    const int64_t seq_len = seq_pos + 1;
    Tensor scores = Tensor::empty({n_heads, seq_len}, DType::F32);
    float* scores_ptr = scores.view.ptr<float>();

    // Access cache memory for K,V
    TensorView Kcache = cache.k_view(); // [L, S, H, D]
    TensorView Vcache = cache.v_view();
    const float* Kbase = Kcache.ptr<const float>();
    const float* Vbase = Vcache.ptr<const float>();

    const int64_t L = cache.config().num_layers;
    const int64_t S = cache.config().max_seq_len;
    const int64_t H = cache.config().num_heads;
    const int64_t D = cache.config().head_dim;

    const int64_t stride_L = S * H * D;
    const int64_t stride_S = H * D;
    const int64_t stride_H = D;
    const int64_t layer_base = layer_idx * stride_L;

    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));

    const float* qptr = q_rot_v.ptr<const float>();
    for (int64_t h = 0; h < n_heads; ++h) {
        const float* qh = qptr + h * d_head;
        for (int64_t t = 0; t < seq_len; ++t) {
            const float* kvec = Kbase + layer_base + t * stride_S + h * stride_H;
            float dot = 0.0f;
            for (int64_t d = 0; d < d_head; ++d) {
                dot += qh[d] * kvec[d];
            }
            scores_ptr[h * seq_len + t] = dot * scale;
        }
    }

    // Step 5: softmax over time dim per head
    Tensor attn = ie::ops::softmax(scores.view, -1);
    const float* attn_ptr = attn.view.ptr<const float>();

    // Step 6: context = attn @ V (manual matmul)
    Tensor ctx = Tensor::empty({n_heads, d_head}, DType::F32);
    float* ctx_ptr = ctx.view.ptr<float>();
    const float* Vb = Vcache.ptr<const float>();
    for (int64_t h = 0; h < n_heads; ++h) {
        float* ch = ctx_ptr + h * d_head;
        for (int64_t d = 0; d < d_head; ++d) ch[d] = 0.0f;
        for (int64_t t = 0; t < seq_len; ++t) {
            float a = attn_ptr[h * seq_len + t];
            const float* vvec = Vb + layer_base + t * stride_S + h * stride_H;
            for (int64_t d = 0; d < d_head; ++d) {
                ch[d] += a * vvec[d];
            }
        }
    }

    // Step 7: output projection: flatten context and apply Wo
    TensorView ctx_flat = make_view(ctx.view.data, ctx.view.dt, {1, n_heads * d_head});
    Tensor out = ie::ops::linear(ctx_flat, weights.Wo, weights.bo);
    return out;
}

} // namespace layers
} // namespace ie
