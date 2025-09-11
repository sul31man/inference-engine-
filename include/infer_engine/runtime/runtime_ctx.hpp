#pragma once
#include "infer_engine/model/config.hpp"
#include "infer_engine/model/weights.hpp"
#include "infer_engine/runtime/kv_cache.hpp"
#include "infer_engine/core/tensor.hpp"
#include <memory>

namespace ie {

class RuntimeCtx {
public:
    RuntimeCtx(const ModelCfg& cfg, const ModelWeights& weights);

    // Forward one decode step: input token_id at position pos -> logits [vocab_size]
    Tensor forward_decode(int32_t token_id, int64_t pos);

    // Accessors
    const ModelCfg& cfg() const { return cfg_; }
    KVCache& kv() { return *kv_; }

private:
    ModelCfg cfg_;
    ModelWeights weights_;
    std::unique_ptr<KVCache> kv_;
};

} // namespace ie


