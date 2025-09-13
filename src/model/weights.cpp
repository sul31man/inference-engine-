#include "infer_engine/model/weights.hpp"
#include <stdexcept>

namespace ie {

TensorView ModelWeights::get_token_embeddings() const {
    if (!token_embeddings_.defined()) {
        throw std::logic_error("token embeddings not bound");
    }
    return token_embeddings_;
}

TensorView ModelWeights::get_lm_head() const {
    if (!lm_head_.defined()) {
        throw std::logic_error("lm head not bound");
    }
    return lm_head_;
}

TensorView ModelWeights::get_final_norm() const {
    if (!final_norm_.defined()) {
        throw std::logic_error("final norm not bound");
    }
    return final_norm_;
}

LayerWeightsCXX ModelWeights::get_layer_weights(int64_t layer_idx) const {
    if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layers_.size()) {
        throw std::out_of_range("get_layer_weights: layer_idx out of range");
    }
    return layers_[static_cast<size_t>(layer_idx)];
}

} // namespace ie

