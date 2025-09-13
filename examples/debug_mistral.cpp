#include "infer_engine/io/model_loader.hpp"
#include "infer_engine/model/config.hpp"
#include "infer_engine/model/weights.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    using namespace ie;
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_mistral_model_dir>\n";
        return 1;
    }
    
    std::string model_dir = argv[1];
    
    try {
        std::cout << "Loading Mistral model from: " << model_dir << std::endl;
        
        ModelCfg cfg;
        ModelWeights weights;
        
        load_mistral_safetensors(model_dir, cfg, weights);
        
        std::cout << "Model configuration:" << std::endl;
        std::cout << "  d_model: " << cfg.d_model << std::endl;
        std::cout << "  n_layers: " << cfg.n_layers << std::endl;
        std::cout << "  n_heads: " << cfg.n_heads << std::endl;
        std::cout << "  vocab_size: " << cfg.vocab_size << std::endl;
        
        // Check critical weight shapes
        auto token_emb = weights.get_token_embeddings();
        std::cout << "\nToken embeddings shape: [";
        for (size_t i = 0; i < token_emb.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << token_emb.shape[i];
        }
        std::cout << "] (expected: [" << cfg.vocab_size << ", " << cfg.d_model << "])" << std::endl;
        
        auto lm_head = weights.get_lm_head();
        std::cout << "LM head shape: [";
        for (size_t i = 0; i < lm_head.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << lm_head.shape[i];
        }
        std::cout << "] (expected: [" << cfg.vocab_size << ", " << cfg.d_model << "])" << std::endl;
        
        // Check first layer weights
        if (weights.num_layers() > 0) {
            auto layer0 = weights.get_layer_weights(0);
            
            std::cout << "\nLayer 0 attention weights:" << std::endl;
            std::cout << "  Q projection: [";
            for (size_t i = 0; i < layer0.attn.Wq.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << layer0.attn.Wq.shape[i];
            }
            std::cout << "] (expected: [" << cfg.d_model << ", " << cfg.d_model << "])" << std::endl;
            
            std::cout << "  K projection: [";
            for (size_t i = 0; i < layer0.attn.Wk.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << layer0.attn.Wk.shape[i];
            }
            std::cout << "]" << std::endl;
            
            std::cout << "  V projection: [";
            for (size_t i = 0; i < layer0.attn.Wv.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << layer0.attn.Wv.shape[i];
            }
            std::cout << "]" << std::endl;
            
            std::cout << "  O projection: [";
            for (size_t i = 0; i < layer0.attn.Wo.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << layer0.attn.Wo.shape[i];
            }
            std::cout << "]" << std::endl;
            
            std::cout << "\nLayer 0 MLP weights:" << std::endl;
            std::cout << "  Gate projection: [";
            for (size_t i = 0; i < layer0.mlp.W1.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << layer0.mlp.W1.shape[i];
            }
            std::cout << "]" << std::endl;
            
            std::cout << "  Up projection: [";
            for (size_t i = 0; i < layer0.mlp.W3.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << layer0.mlp.W3.shape[i];
            }
            std::cout << "]" << std::endl;
            
            std::cout << "  Down projection: [";
            for (size_t i = 0; i < layer0.mlp.W2.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << layer0.mlp.W2.shape[i];
            }
            std::cout << "]" << std::endl;
        }
        
        // Test simple token embedding lookup
        std::cout << "\nTesting token embedding lookup..." << std::endl;
        int32_t test_token = 72; // 'H'
        
        if (test_token < cfg.vocab_size) {
            const float* embed_ptr = token_emb.ptr<float>() + test_token * token_emb.shape[1];
            std::cout << "Token " << test_token << " embedding (first 5 values): ";
            for (int i = 0; i < 5 && i < token_emb.shape[1]; ++i) {
                std::cout << embed_ptr[i] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nDebug completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
