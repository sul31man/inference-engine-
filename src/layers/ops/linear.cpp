#include "infer_engine/layers/ops/linear.hpp"
#include "infer_engine/core/tensor.hpp"

namespace ie {
namespace ops {

Tensor linear(const TensorView& x, const TensorView& W, const TensorView* bias) {
    // Shapes:
    // x: [N, D_in]
    // W: [D_out, D_in] 
    // bias: [D_out] (optional)
    // output: [N, D_out]
    int64_t N = x.shape[0];
    int64_t D_in = x.shape[1];
    int64_t D_out = W.shape[0];

    auto output = Tensor::empty({N, D_out}, x.dt);

    // Transpose W to get W_t [D_in, D_out]
    auto W_t = Tensor::empty({D_in, D_out}, W.dt);

    float* Wt_ptr = W_t.view.ptr<float>();
    const float* W_ptr = W.ptr<const float>();
    const float* x_ptr = x.ptr<const float>();

    int64_t W_rows = W.shape[0];
    int64_t W_cols = W.shape[1];

    for(int64_t i = 0; i < W_cols; i++){
        for(int64_t j = 0; j < W_rows; j++){
            Wt_ptr[i*W_rows + j] = W_ptr[j*W_cols + i];
        }
    }

    float* output_ptr = output.view.ptr<float>();

    // y = x @ W.T → using W_t
    for (int64_t i = 0; i < N; i++){
        for(int64_t j = 0; j < D_out; j++){
            float value = 0.0f;
            for(int64_t k = 0; k < D_in; k++){
                value += x_ptr[i*D_in + k] * Wt_ptr[k*D_out + j];
            }
            output_ptr[i*D_out + j] = value;
        }
    }

    if (bias) {
        const float* bias_ptr = bias->ptr<const float>();
        for (int64_t i = 0; i < N; i++) {
            for (int64_t j = 0; j < D_out; j++) {
                output_ptr[i*D_out + j] += bias_ptr[j];
            }
        }
    }

    return output;
}

} // namespace ops
} // namespace ie
