#include "infer_engine/io/model_loader.hpp"
#include "infer_engine/model/config.hpp"
#include "infer_engine/model/weights.hpp"
#include "infer_engine/runtime/runtime_ctx.hpp"
#include "infer_engine/core/tensor.hpp"
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <random>

int main(int argc, char** argv) {
    using namespace ie;
    
    // Get model directory and prompt from command line
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_mistral_model_dir> [prompt]\n";
        std::cerr << "Example: " << argv[0] << " ~/.cache/huggingface/hub/models--mlx-community--Mistral-7B-Instruct-v0.3-4bit/snapshots/<hash>/ \"Hello, how are you?\"\n";
        return 1;
    }
    
    std::string model_dir = argv[1];
    std::string prompt = (argc > 2) ? std::string(argv[2]) : std::string("Hello, how are you?");
    
    try {
        std::cout << "Loading Mistral model from: " << model_dir << std::endl;
        
        ModelCfg cfg;
        ModelWeights weights;
        
        // Load the actual Mistral model
        load_mistral_safetensors(model_dir, cfg, weights);
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Model info: " << cfg.n_layers << " layers, " << cfg.d_model << " hidden size" << std::endl;
        
        // Initialize runtime context with the loaded model
        RuntimeCtx ctx(cfg, weights);
        
        std::cout << "Runtime context initialized!" << std::endl;
        std::cout << "Running inference with prompt: \"" << prompt << "\"" << std::endl;
        
        // Simple tokenization (byte-level for demo)
        std::vector<int32_t> input_ids;
        for (char c : prompt) {
            input_ids.push_back(static_cast<int32_t>(static_cast<unsigned char>(c)));
        }
        
        std::cout << "Input token IDs: [";
        for (size_t i = 0; i < input_ids.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << input_ids[i];
        }
        std::cout << "]" << std::endl;
        
        // Process prompt tokens one by one (prefill phase)
        std::cout << "Processing prompt tokens..." << std::endl;
        Tensor last_logits;
        
        for (size_t i = 0; i < input_ids.size(); ++i) {
            int32_t token_id = input_ids[i];
            int64_t pos = static_cast<int64_t>(i);
            
            std::cout << "Processing token " << i << "/" << input_ids.size() << " (id=" << token_id << ")" << std::endl;
            
            last_logits = ctx.forward_decode(token_id, pos);
        }
        
        std::cout << "Prompt processing completed!" << std::endl;
        std::cout << "Logits shape: [";
        for (size_t i = 0; i < last_logits.view.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << last_logits.view.shape[i];
        }
        std::cout << "]" << std::endl;
        
        // Simple greedy decoding for next token prediction
        int64_t vocab_size = last_logits.view.shape[0];
        const float* logits = last_logits.view.ptr<float>();
        
        // Find token with highest probability
        int32_t next_token = 0;
        float max_logit = logits[0];
        for (int64_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                next_token = static_cast<int32_t>(i);
            }
        }
        
        std::cout << "Next predicted token ID: " << next_token << std::endl;
        if (next_token >= 32 && next_token < 127) {  // Printable ASCII range
            std::cout << "Next predicted token (as char): '" << static_cast<char>(next_token) << "'" << std::endl;
        }
        std::cout << "Confidence (logit): " << max_logit << std::endl;
        
        // Generate a few more tokens
        std::cout << "\nGenerating sequence:" << std::endl;
        std::cout << "\"" << prompt;
        
        std::vector<int32_t> generated_tokens;
        int32_t current_token = next_token;
        
        for (int step = 0; step < 10; ++step) {  // Generate 10 more tokens
            // Add current token to output
            generated_tokens.push_back(current_token);
            if (current_token >= 32 && current_token < 127) {
                std::cout << static_cast<char>(current_token);
            } else {
                std::cout << "[" << current_token << "]";
            }
            std::cout.flush();
            
            // Generate next token
            int64_t next_pos = static_cast<int64_t>(input_ids.size() + step + 1);
            Tensor next_logits = ctx.forward_decode(current_token, next_pos);
            
            // Greedy sampling
            const float* next_logits_ptr = next_logits.view.ptr<float>();
            int32_t predicted_token = 0;
            float max_next_logit = next_logits_ptr[0];
            for (int64_t i = 1; i < vocab_size; ++i) {
                if (next_logits_ptr[i] > max_next_logit) {
                    max_next_logit = next_logits_ptr[i];
                    predicted_token = static_cast<int32_t>(i);
                }
            }
            
            current_token = predicted_token;
            
            // Stop on certain tokens (simple stopping criteria)
            if (current_token == 0 || current_token == 10 || current_token == 13) {  // null, LF, CR
                break;
            }
        }
        
        std::cout << "\"" << std::endl;
        
        std::cout << "\nInference completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
