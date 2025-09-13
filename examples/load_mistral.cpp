#include "infer_engine/io/model_loader.hpp"
#include "infer_engine/model/config.hpp"
#include "infer_engine/model/weights.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    using namespace ie;
    
    // Get model directory path from command line
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_mistral_model_dir>\n";
        std::cerr << "Example: " << argv[0] << " ~/.cache/huggingface/hub/models--mlx-community--Mistral-7B-Instruct-v0.3-4bit/snapshots/<hash>/\n";
        return 1;
    }
    
    std::string model_dir = argv[1];
    
    try {
        std::cout << "Loading Mistral model from: " << model_dir << std::endl;
        
        ModelCfg cfg;
        ModelWeights weights;
        
        // Load the model
        load_mistral_safetensors(model_dir, cfg, weights);
        
        std::cout << "Successfully loaded Mistral model!" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  d_model: " << cfg.d_model << std::endl;
        std::cout << "  n_layers: " << cfg.n_layers << std::endl;
        std::cout << "  n_heads: " << cfg.n_heads << std::endl;
        std::cout << "  n_kv_heads: " << cfg.n_kv_heads << std::endl;
        std::cout << "  vocab_size: " << cfg.vocab_size << std::endl;
        std::cout << "  rope_theta: " << cfg.rope_theta << std::endl;
        std::cout << "  rope_dim: " << cfg.rope_dim << std::endl;
        
        // Test access to various weights
        std::cout << "\nWeight tensor shapes:" << std::endl;
        
        auto token_emb = weights.get_token_embeddings();
        std::cout << "  Token embeddings: [";
        for (size_t i = 0; i < token_emb.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << token_emb.shape[i];
        }
        std::cout << "]" << std::endl;
        
        auto lm_head = weights.get_lm_head();
        std::cout << "  LM head: [";
        for (size_t i = 0; i < lm_head.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << lm_head.shape[i];
        }
        std::cout << "]" << std::endl;
        
        auto final_norm = weights.get_final_norm();
        std::cout << "  Final norm: [";
        for (size_t i = 0; i < final_norm.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << final_norm.shape[i];
        }
        std::cout << "]" << std::endl;
        
        // Test first layer weights
        if (weights.num_layers() > 0) {
            auto layer0 = weights.get_layer_weights(0);
            std::cout << "  Layer 0 Q projection: [";
            for (size_t i = 0; i < layer0.attn.Wq.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << layer0.attn.Wq.shape[i];
            }
            std::cout << "]" << std::endl;
            
            if (layer0.input_layernorm) {
                std::cout << "  Layer 0 input norm: [";
                for (size_t i = 0; i < layer0.input_layernorm->shape.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << layer0.input_layernorm->shape[i];
                }
                std::cout << "]" << std::endl;
            }
        }
        
        std::cout << "\nModel loaded successfully! You can now use it for inference." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
