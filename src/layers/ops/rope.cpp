#include "infer_engine/layers/ops/rope.hpp"
#include "infer_engine/core/tensor.hpp"
#include <cmath>
#include <tuple>

namespace ie {
namespace ops {

std::tuple<Tensor, Tensor> rope_apply(
    const TensorView& q, 
    const TensorView& k, 
    const TensorView& pos,
    int rotary_dim,
    float theta_base
) {
    // Simple RoPE assuming last dim contains rotary and non-rotary parts.
    // Here we expect `pos` to already contain cos/sin pairs per position and pair-dimension:
    // pos shape [..., rotary_dim/2, 2] with [..., 0]=cos and [..., 1]=sin.

    int64_t D = q.shape.back();
    int64_t use_dim = (rotary_dim <= 0) ? D : rotary_dim;
    int64_t pairs = use_dim / 2;

    auto q_out_t = Tensor::empty(q.shape, q.dt);
    auto k_out_t = Tensor::empty(k.shape, k.dt);

    const float* q_ptr = q.ptr<const float>();
    const float* k_ptr = k.ptr<const float>();
    float* q_out = q_out_t.view.ptr<float>();
    float* k_out = k_out_t.view.ptr<float>();

    // Flattened processing: process per last-dim vector
    int64_t groups = q.numel() / D;

    // For this skeleton, assume pos provides cos/sin per group and pair
    // with layout: pos[..., pairs, 2]. We'll index pos linearly as well.
    const float* pos_ptr = pos.ptr<const float>();
    int64_t pos_stride = pairs * 2; // per-group

    for (int64_t g = 0; g < groups; ++g) {
        const float* qv = q_ptr + g * D;
        const float* kv = k_ptr + g * D;
        float* qvo = q_out + g * D;
        float* kvo = k_out + g * D;

        const float* p = pos_ptr + g * pos_stride;

        // rotate first `use_dim` dims as pairs
        for (int64_t i = 0; i < pairs; ++i) {
            float xq = qv[2*i + 0];
            float yq = qv[2*i + 1];
            float xk = kv[2*i + 0];
            float yk = kv[2*i + 1];

            float c = p[2*i + 0];
            float s = p[2*i + 1];

            qvo[2*i + 0] = xq * c - yq * s;
            qvo[2*i + 1] = xq * s + yq * c;
            kvo[2*i + 0] = xk * c - yk * s;
            kvo[2*i + 1] = xk * s + yk * c;
        }

        // copy the remaining dims if any
        for (int64_t i = use_dim; i < D; ++i) {
            qvo[i] = qv[i];
            kvo[i] = kv[i];
        }
    }

    return std::make_tuple(std::move(q_out_t), std::move(k_out_t));
}

} // namespace ops
} // namespace ie
