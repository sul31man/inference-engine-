#include "infer_engine/io/model_loader.hpp"
#include "infer_engine/io/safetensors_reader.hpp"
#include "infer_engine/model/config.hpp"
#include "infer_engine/model/weights.hpp"
#include "infer_engine/core/tensor.hpp"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace ie {

static Tensor load_bin_tensor(const std::string& path, const std::vector<int64_t>& shape) {
    size_t numel = 1; for (auto d : shape) numel *= static_cast<size_t>(d);
    size_t bytes = numel * sizeof(float);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open bin: " + path);
    Tensor t = Tensor::empty(shape, DType::F32);
    f.read(reinterpret_cast<char*>(t.view.ptr<float>()), bytes);
    if (!f) throw std::runtime_error("Failed to read bin: " + path);
    return t;
}

void load_tiny_bins(const std::string& dir, ModelCfg& cfg, ModelWeights& weights) {
    // Read model.json
    // TODO: use a real JSON parser; for now, expect Python created values and set manually
    // Placeholder: try to open and parse minimally
    // You can replace this with load_cfg when implemented

    // Bind embeddings and head
    Tensor tok_emb = load_bin_tensor(dir + "/tok_emb.bin", {cfg.vocab_size, cfg.d_model});
    Tensor lm_head = load_bin_tensor(dir + "/lm_head.bin", {cfg.vocab_size, cfg.d_model});
    weights.set_token_embeddings(tok_emb.view);
    weights.set_lm_head(lm_head.view);

    weights.set_num_layers(cfg.n_layers);
    int64_t d_head = cfg.d_model / cfg.n_heads;
    int64_t d_ff = 4 * cfg.d_model;
    for (int64_t L = 0; L < cfg.n_layers; ++L) {
        LayerWeightsCXX lw;
        lw.attn.Wq = load_bin_tensor(dir + "/Wq_" + std::to_string(L) + ".bin", {cfg.d_model, cfg.d_model}).view;
        lw.attn.Wk = load_bin_tensor(dir + "/Wk_" + std::to_string(L) + ".bin", {cfg.d_model, cfg.d_model}).view;
        lw.attn.Wv = load_bin_tensor(dir + "/Wv_" + std::to_string(L) + ".bin", {cfg.d_model, cfg.d_model}).view;
        lw.attn.Wo = load_bin_tensor(dir + "/Wo_" + std::to_string(L) + ".bin", {cfg.d_model, cfg.d_model}).view;
        lw.mlp.W1 = load_bin_tensor(dir + "/W1_" + std::to_string(L) + ".bin", {cfg.d_model, d_ff}).view;
        lw.mlp.W3 = load_bin_tensor(dir + "/W3_" + std::to_string(L) + ".bin", {cfg.d_model, d_ff}).view;
        lw.mlp.W2 = load_bin_tensor(dir + "/W2_" + std::to_string(L) + ".bin", {d_ff, cfg.d_model}).view;
        weights.set_layer_weights(L, lw);
    }
}

// Simple JSON parser for config.json
static void parse_mistral_config(const std::string& config_path, ModelCfg& cfg) {
    std::ifstream f(config_path);
    if (!f) {
        throw std::runtime_error("Failed to open config.json: " + config_path);
    }
    
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    
    // Simple parsing for Mistral config
    auto find_int_value = [&](const std::string& key) -> int64_t {
        std::string search = "\"" + key + "\"";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return 0;
        
        size_t colon = content.find(':', pos);
        if (colon == std::string::npos) return 0;
        
        size_t start = colon + 1;
        while (start < content.length() && (content[start] == ' ' || content[start] == '\t')) start++;
        
        size_t end = start;
        while (end < content.length() && std::isdigit(content[end])) end++;
        
        if (start == end) return 0;
        return std::stoll(content.substr(start, end - start));
    };
    
    auto find_float_value = [&](const std::string& key) -> float {
        std::string search = "\"" + key + "\"";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return 0.0f;
        
        size_t colon = content.find(':', pos);
        if (colon == std::string::npos) return 0.0f;
        
        size_t start = colon + 1;
        while (start < content.length() && (content[start] == ' ' || content[start] == '\t')) start++;
        
        size_t end = start;
        while (end < content.length() && (std::isdigit(content[end]) || content[end] == '.' || content[end] == 'e' || content[end] == '-' || content[end] == '+')) end++;
        
        if (start == end) return 0.0f;
        return std::stof(content.substr(start, end - start));
    };
    
    cfg.d_model = find_int_value("hidden_size");
    cfg.n_layers = find_int_value("num_hidden_layers");
    cfg.n_heads = find_int_value("num_attention_heads");
    cfg.n_kv_heads = find_int_value("num_key_value_heads");  // GQA support
    cfg.vocab_size = find_int_value("vocab_size");
    cfg.rope_theta = find_float_value("rope_theta");
    cfg.rope_dim = cfg.d_model / cfg.n_heads;  // Standard for Mistral
}

// No global upcast: keep weights as stored (BF16/F16/F32). Matmuls upcast on-the-fly.

void load_mistral_safetensors(const std::string& model_dir, ModelCfg& cfg, ModelWeights& weights) {
    // Parse config.json
    std::string config_path = model_dir + "/config.json";
    parse_mistral_config(config_path, cfg);
    
    // Open safetensors file (try consolidated first, then fall back to model.safetensors)
    std::string safetensors_path = model_dir + "/consolidated.safetensors";
    std::ifstream test_file(safetensors_path);
    if (!test_file.good()) {
        safetensors_path = model_dir + "/model.safetensors";
    }
    test_file.close();
    
    // Keep reader alive by shared_ptr captured into weights
    auto reader_sp = std::make_shared<SafeTensorReader>(safetensors_path);
    
    // Load embeddings (keep source dtype)
    {
        TensorView tv = reader_sp->get_tensor("tok_embeddings.weight");
        weights.set_token_embeddings(tv);
    }
    
    // Load language model head (keep source dtype)
    {
        TensorView tv = reader_sp->get_tensor("output.weight");
        weights.set_lm_head(tv);
    }
    
    // Load final norm (keep source dtype)
    {
        TensorView tv = reader_sp->get_tensor("norm.weight");
        weights.set_final_norm(tv);
    }
    
    // Initialize layer storage
    weights.set_num_layers(cfg.n_layers);
    
    // Bind layer tensors directly from storage (no global upcast)
    
    // Load each layer
    for (int64_t layer_idx = 0; layer_idx < cfg.n_layers; ++layer_idx) {
        LayerWeightsCXX layer_weights;
        std::string layer_prefix = "layers." + std::to_string(layer_idx) + ".";
        
        // Attention weights
        layer_weights.attn.Wq = reader_sp->get_tensor(layer_prefix + "attention.wq.weight");
        layer_weights.attn.Wk = reader_sp->get_tensor(layer_prefix + "attention.wk.weight");
        layer_weights.attn.Wv = reader_sp->get_tensor(layer_prefix + "attention.wv.weight");
        layer_weights.attn.Wo = reader_sp->get_tensor(layer_prefix + "attention.wo.weight");

        // MLP weights
        TensorView w1 = reader_sp->get_tensor(layer_prefix + "feed_forward.w1.weight");
        TensorView w2 = reader_sp->get_tensor(layer_prefix + "feed_forward.w2.weight");
        layer_weights.mlp.W1 = w1;            // gate/first
        layer_weights.mlp.W3 = w1;            // reuse if only w1/w2 present
        layer_weights.mlp.W2 = w2;            // down

        // Layer norms
        TensorView attn_norm = reader_sp->get_tensor(layer_prefix + "attention_norm.weight");
        TensorView ffn_norm = reader_sp->get_tensor(layer_prefix + "ffn_norm.weight");
        static std::vector<TensorView> attn_norm_views; // persistent holders not needed: we store pointers
        layer_weights.input_layernorm = new TensorView(attn_norm);
        layer_weights.post_attention_layernorm = new TensorView(ffn_norm);
        
        weights.set_layer_weights(layer_idx, layer_weights);
    }
    // Capture owner to keep memory-mapped data alive
    weights.set_owner(reader_sp);
}

} // namespace ie


