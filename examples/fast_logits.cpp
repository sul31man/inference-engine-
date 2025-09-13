#include "infer_engine/io/safetensors_reader.hpp"
#include "infer_engine/model/config.hpp"
#include "infer_engine/core/tensor.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>

// Local BF16->F32 helpers (reuse existing converter signatures if available)
namespace ie { Tensor convert_bf16_to_f32(const TensorView& bf16_tensor); }

// Very simple config parser tailored to Mistral's config.json
static int64_t parse_int_field(const std::string& content, const std::string& key) {
    std::string k = "\"" + key + "\"";
    size_t pos = content.find(k);
    if (pos == std::string::npos) return 0;
    size_t colon = content.find(':', pos);
    if (colon == std::string::npos) return 0;
    size_t start = colon + 1;
    while (start < content.size() && (content[start] == ' ' || content[start] == '\t')) start++;
    size_t end = start;
    while (end < content.size() && (isdigit(content[end]) || content[end] == '-')) end++;
    if (start == end) return 0;
    return std::stoll(content.substr(start, end - start));
}

int main(int argc, char** argv) {
    using namespace ie;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mistral_model_dir> [token_id]" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    int64_t token_id = (argc >= 3) ? std::stoll(argv[2]) : 0;

    try {
        // Read config for dims
        std::ifstream cf(model_dir + "/config.json");
        if (!cf) { std::cerr << "Failed to open config.json" << std::endl; return 1; }
        std::string cfg_content((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        int64_t d_model = parse_int_field(cfg_content, "hidden_size");
        int64_t vocab_size = parse_int_field(cfg_content, "vocab_size");

        // Open safetensors (prefer consolidated)
        std::string st_path = model_dir + "/consolidated.safetensors";
        std::ifstream tf(st_path); if (!tf.good()) st_path = model_dir + "/model.safetensors";
        SafeTensorReader reader(st_path);

        // Load just embeddings and lm_head
        TensorView emb_tv = reader.get_tensor("tok_embeddings.weight");   // [vocab, d_model]
        TensorView head_tv = reader.get_tensor("output.weight");         // [vocab, d_model]

        // Convert to F32 if BF16
        Tensor emb = (emb_tv.dt == DType::F32) ? Tensor::from_raw(emb_tv.ptr<void>(), emb_tv.shape, DType::F32)
                                               : convert_bf16_to_f32(emb_tv);
        Tensor head = (head_tv.dt == DType::F32) ? Tensor::from_raw(head_tv.ptr<void>(), head_tv.shape, DType::F32)
                                                 : convert_bf16_to_f32(head_tv);

        if (token_id < 0 || token_id >= vocab_size) {
            std::cerr << "token_id out of range (0.." << (vocab_size-1) << ")\n";
            return 1;
        }

        // Fetch embedding row for token
        const float* emb_ptr = emb.view.ptr<const float>() + token_id * d_model;

        // logits = head @ emb; head is [vocab, d_model]
        std::vector<float> logits(vocab_size, 0.0f);
        const float* head_ptr = head.view.ptr<const float>();
        for (int64_t v = 0; v < vocab_size; ++v) {
            const float* w = head_ptr + v * d_model;
            float s = 0.0f;
            for (int64_t d = 0; d < d_model; ++d) s += w[d] * emb_ptr[d];
            logits[static_cast<size_t>(v)] = s;
        }

        // Print top-5 indices
        std::vector<int64_t> idx(vocab_size);
        for (int64_t i = 0; i < vocab_size; ++i) idx[static_cast<size_t>(i)] = i;
        std::partial_sort(idx.begin(), idx.begin()+5, idx.end(), [&](int64_t a, int64_t b){ return logits[(size_t)a] > logits[(size_t)b]; });

        std::cout << "Top-5 logits for token_id=" << token_id << ":\n";
        for (int i = 0; i < 5; ++i) {
            int64_t v = idx[(size_t)i];
            std::cout << "  " << i+1 << ") id=" << v << ", logit=" << logits[(size_t)v] << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

