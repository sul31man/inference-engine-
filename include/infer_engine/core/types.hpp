#pragma once 
#include <cstdint> 
#include <stdexcept> 
#include <string> 


namespace ie{

    //Basic status macro 

enum class DType : uint8_t { F32 = 0, F16 = 1, BF16 = 2, I8 = 3};

inline const char* dtype_name(DType dt){

    switch(dt){ case DType::F32: return "f32"; case DType::F16: return "f16"; case DType::BF16: return "BF16"; case DType::I8: return "I8"; }
    return "unknown";
}

inline size_t dtype_bytes(DType dt){

    switch(dt){ case DType::F32: return 4; case DType::F16: return 2; 
                case DType::BF16: return 2; case DType::I8: return 1;}
    throw std::runtime_error("bad dtype");
}

struct Shape {

    std::vector<int64_t> dims; 
    int64_t rank() const {return (int64_t)dims.size();}
    int64_t numel() const {

        int64_t n=1; for(auto d:dims) n *=d; return n; 
    }
};


};





