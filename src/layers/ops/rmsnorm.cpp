#include "infer_engine/layers/ops/rmsnorm.hpp"
#include "infer_engine/core/tensor.hpp"
#include <cmath>

namespace ie {
namespace ops {

Tensor rmsnorm(const TensorView& x, const TensorView& gamma, float eps) {
    // Normalize along the last dimension for all leading dims
    auto output = Tensor::empty(x.shape, x.dt);

    const float* input_ptr = x.ptr<const float>();
    const float* gamma_ptr = gamma.ptr<const float>();
    float* output_ptr = output.view.ptr<float>();

    int64_t D = x.shape.back();
    int64_t groups = x.numel() / D; // number of vectors along last dim

    for (int64_t g = 0; g < groups; ++g) {
        const float* in = input_ptr + g * D;
        float* out = output_ptr + g * D;

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < D; ++i) {
            float v = in[i];
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(D) + eps);
        float inv_rms = 1.0f / rms;

        for (int64_t i = 0; i < D; ++i) {
            out[i] = (in[i] * inv_rms) * gamma_ptr[i];
        }
    }

    return output;
}

} // namespace ops
} // namespace ie
