#include "infer_engine/io/safetensors_reader.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    using namespace ie;
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_safetensors_file>\n";
        return 1;
    }
    
    std::string safetensors_path = argv[1];
    
    try {
        std::cout << "Opening safetensors file: " << safetensors_path << std::endl;
        
        SafeTensorReader reader(safetensors_path);
        
        // Get all tensor names
        auto tensor_names = reader.get_tensor_names();
        
        std::cout << "Found " << tensor_names.size() << " tensors:" << std::endl;
        std::cout << "================================" << std::endl;
        
        // Show first 20 tensors
        size_t count = 0;
        for (const auto& name : tensor_names) {
            if (count >= 20) break;
            
            auto info = reader.get_tensor_info(name);
            
            std::cout << name << ": [";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << info.shape[i];
            }
            std::cout << "] (" << info.dtype << ")" << std::endl;
            
            count++;
        }
        
        if (tensor_names.size() > 20) {
            std::cout << "... and " << (tensor_names.size() - 20) << " more tensors" << std::endl;
        }
        
        // Look for embedding and lm_head specifically
        std::cout << "\n=== Looking for key tensors ===" << std::endl;
        
        std::vector<std::string> key_patterns = {
            "embed", "embedding", "lm_head", "output", "norm", "0.weight"
        };
        
        for (const auto& pattern : key_patterns) {
            std::cout << "\nTensors containing '" << pattern << "':" << std::endl;
            bool found = false;
            for (const auto& name : tensor_names) {
                if (name.find(pattern) != std::string::npos) {
                    auto info = reader.get_tensor_info(name);
                    std::cout << "  " << name << ": [";
                    for (size_t i = 0; i < info.shape.size(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << info.shape[i];
                    }
                    std::cout << "]" << std::endl;
                    found = true;
                }
            }
            if (!found) {
                std::cout << "  (none found)" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
