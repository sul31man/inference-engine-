#include "infer_engine/io/model_loader.hpp"
#include "infer_engine/model/config.hpp"
#include "infer_engine/model/weights.hpp"
#include "infer_engine/runtime/runtime_ctx.hpp"
#include "infer_engine/core/tensor.hpp"
#include <iostream>

int main(int argc, char** argv) {
    using namespace ie;
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_mistral_model_dir>" << std::endl;
        return 1;
    }
    
    std::string model_dir = argv[1];
    
    try {
        std::cout << "Loading Mistral model..." << std::endl;
        
        ModelCfg cfg;
        ModelWeights weights;
        load_mistral_safetensors(model_dir, cfg, weights);
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Config: n_heads=" << cfg.n_heads << ", n_kv_heads=" << cfg.n_kv_heads << std::endl;
        
        // Test 1: Try to create RuntimeCtx
        std::cout << "Creating runtime context..." << std::endl;
        RuntimeCtx runtime(cfg, weights);
        std::cout << "Runtime context created successfully!" << std::endl;
        
        // Test 2: Try a simple forward pass with a dummy token
        std::cout << "Testing forward pass with token 0..." << std::endl;
        
        try {
            Tensor result = runtime.forward_decode(0, 0);  // token_id=0, position=0
            std::cout << "Forward pass successful! Output shape: [" << result.view.shape[0] << "]" << std::endl;
            
            // Show first few logits
            const float* logits = result.view.ptr<float>();
            std::cout << "First 5 logits: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << logits[i] << " ";
            }
            std::cout << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Forward pass failed: " << e.what() << std::endl;
            return 1;
        }
        
        std::cout << "All tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
