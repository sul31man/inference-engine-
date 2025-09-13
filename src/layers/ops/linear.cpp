#include "infer_engine/layers/ops/linear.hpp"
#include "infer_engine/core/tensor.hpp"
#include <cassert>
#include <cstdint>
#ifdef IE_OMP
#include <omp.h>
#endif

namespace ie {
namespace ops {

static inline float bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } out;
    out.u = static_cast<uint32_t>(h) << 16;
    return out.f;
}

static inline float f16_to_f32(uint16_t h) {
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
}

Tensor linear(const TensorView& x, const TensorView& W, const TensorView* bias) {
    // x: [N, D_in] or [D_in]
    // W: [D_out, D_in]
    int64_t N = (x.shape.size() == 1) ? 1 : x.shape[0];
    int64_t D_in = (x.shape.size() == 1) ? x.shape[0] : x.shape[1];
    assert(W.shape.size() == 2 && "W must be rank-2 [D_out, D_in]");
    int64_t D_out = W.shape[0];
    int64_t W_Din = W.shape[1];
    assert(W_Din == D_in && "W.shape[1] must equal D_in");

    // Always accumulate/output in F32
    auto output = Tensor::empty({N, D_out}, DType::F32);
    float* y = output.view.ptr<float>();

    // Read x as F32 (convert if needed per element)
    auto load_x = [&](int64_t idx) -> float {
        if (x.dt == DType::F32) return x.ptr<const float>()[idx];
        if (x.dt == DType::BF16) return bf16_to_f32(x.ptr<const uint16_t>()[idx]);
        if (x.dt == DType::F16) return f16_to_f32(x.ptr<const uint16_t>()[idx]);
        // Default, treat as F32
        return x.ptr<const float>()[idx];
    };

    // Read W row element, dt-aware
    auto load_w = [&](int64_t row, int64_t col) -> float {
        if (W.dt == DType::F32) return W.ptr<const float>()[row * W_Din + col];
        if (W.dt == DType::BF16) return bf16_to_f32(W.ptr<const uint16_t>()[row * W_Din + col]);
        if (W.dt == DType::F16) return f16_to_f32(W.ptr<const uint16_t>()[row * W_Din + col]);
        return W.ptr<const float>()[row * W_Din + col];
    };

    for (int64_t i = 0; i < N; ++i) {
        #ifdef IE_OMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t j = 0; j < D_out; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < D_in; ++k) {
                float xv = load_x((x.shape.size() == 1) ? k : i * D_in + k);
                float wv = load_w(j, k);
                acc += xv * wv;
            }
            y[i * D_out + j] = acc;
        }
    }

    if (bias) {
        // Bias assumed F32
        const float* b = bias->ptr<const float>();
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < D_out; ++j) y[i * D_out + j] += b[j];
        }
    }
    return output;
}

} // namespace ops
} // namespace ie
