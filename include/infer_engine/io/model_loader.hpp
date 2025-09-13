#pragma once
#include <string>

namespace ie {
struct ModelCfg;
class ModelWeights;

// Load simple .bin export created by tools/download_tiny_gpt2.py into ModelCfg+ModelWeights.
void load_tiny_bins(const std::string& dir, ModelCfg& cfg, ModelWeights& weights);

// Load Mistral model from HuggingFace safetensors format
void load_mistral_safetensors(const std::string& model_dir, ModelCfg& cfg, ModelWeights& weights);

} // namespace ie


