#include "infer_engine/runtime/runtime_ctx.hpp"
#include "infer_engine/layers/attention_forward.hpp"
#include "infer_engine/layers/mlp_forward.hpp"
#include <stdexcept>

namespace ie {

RuntimeCtx::RuntimeCtx(const ModelCfg& cfg, const ModelWeights& weights)
    : cfg_(cfg), weights_(weights) {
    KVCacheConfig kcfg;
    kcfg.num_layers = cfg_.n_layers;
    kcfg.max_seq_len = 4096; // TODO: set from cfg or caller
    kcfg.num_heads = cfg_.n_heads;
    kcfg.head_dim = cfg_.d_model / cfg_.n_heads;
    kcfg.dtype = DType::F32;
    kv_ = std::make_unique<KVCache>(kcfg);
}

Tensor RuntimeCtx::forward_decode(int32_t token_id, int64_t pos) {
    // TODO: 
    // 1) Lookup token embedding -> x
    // 2) For each layer: attn_forward + mlp_forward
    // 3) Final projection to logits
    // 4) Return logits [vocab_size]
    throw std::logic_error("forward_decode not implemented");
}

} // namespace ie


