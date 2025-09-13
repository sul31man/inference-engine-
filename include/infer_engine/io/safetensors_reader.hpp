#pragma once
#include "infer_engine/core/tensor.hpp"
#include <string>
#include <memory>
#include <map>

namespace ie {

struct SafeTensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_offset;
    size_t data_size;
};

class SafeTensorReader {
public:
    explicit SafeTensorReader(const std::string& filepath);
    ~SafeTensorReader();

    // Get tensor by name, returns TensorView that references memory-mapped data
    TensorView get_tensor(const std::string& name) const;
    
    // Check if tensor exists
    bool has_tensor(const std::string& name) const;
    
    // Get tensor info without loading data
    const SafeTensorInfo& get_tensor_info(const std::string& name) const;
    
    // List all tensor names
    std::vector<std::string> get_tensor_names() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ie
