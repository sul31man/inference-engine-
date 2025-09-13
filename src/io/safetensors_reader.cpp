#include "infer_engine/io/safetensors_reader.hpp"
#include "infer_engine/core/types.hpp"
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <map>

namespace ie {

// Map safetensors dtype to our DType enum
static DType parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "F32") return DType::F32;
    if (dtype_str == "F16") return DType::F16;
    if (dtype_str == "BF16") return DType::BF16;
    if (dtype_str == "I8") return DType::I8;
    // For now, map unsupported types to closest equivalent
    if (dtype_str == "I32" || dtype_str == "I64" || dtype_str == "U8" || dtype_str == "U32") {
        return DType::F32; // Fallback to F32 for quantized weights
    }
    throw std::runtime_error("Unsupported dtype: " + dtype_str);
}

static size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: case DType::BF16: return 2;
        case DType::I8: return 1;
        default: throw std::runtime_error("Unknown dtype size");
    }
}

class SafeTensorReader::Impl {
public:
    int fd_;
    void* mapped_data_;
    size_t file_size_;
    std::map<std::string, SafeTensorInfo> tensor_info_;
    size_t data_section_offset_;

    explicit Impl(const std::string& filepath) {
        // Open file
        fd_ = open(filepath.c_str(), O_RDONLY);
        if (fd_ == -1) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        // Get file size
        struct stat st;
        if (fstat(fd_, &st) == -1) {
            close(fd_);
            throw std::runtime_error("Failed to stat file: " + filepath);
        }
        file_size_ = st.st_size;

        // Memory map the file
        mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped_data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Failed to mmap file: " + filepath);
        }

        parse_header();
    }

    ~Impl() {
        if (mapped_data_ != MAP_FAILED) {
            munmap(mapped_data_, file_size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }
    }

private:
    void parse_header() {
        if (file_size_ < 8) {
            throw std::runtime_error("File too small for safetensors header");
        }

        // Read 8-byte little-endian header length
        const uint8_t* data = static_cast<const uint8_t*>(mapped_data_);
        uint64_t header_len = 0;
        for (int i = 0; i < 8; ++i) {
            header_len |= static_cast<uint64_t>(data[i]) << (i * 8);
        }

        if (8 + header_len > file_size_) {
            throw std::runtime_error("Invalid header length");
        }

        // Parse JSON header (simplified parser for safetensors format)
        std::string header_json(reinterpret_cast<const char*>(data + 8), header_len);
        data_section_offset_ = 8 + header_len;

        parse_json_metadata(header_json);
    }

    void parse_json_metadata(const std::string& json_str) {
        // Simple JSON parser for safetensors metadata
        // Format: {"tensor_name": {"dtype": "F32", "shape": [2,3], "data_offsets": [start, end]}, ...}
        
        size_t pos = 0;
        while (pos < json_str.length()) {
            // Find next tensor entry
            size_t quote_start = json_str.find('"', pos);
            if (quote_start == std::string::npos) break;
            
            size_t quote_end = json_str.find('"', quote_start + 1);
            if (quote_end == std::string::npos) break;
            
            std::string tensor_name = json_str.substr(quote_start + 1, quote_end - quote_start - 1);
            
            // Find the object for this tensor
            size_t obj_start = json_str.find('{', quote_end);
            if (obj_start == std::string::npos) break;
            
            // Find matching closing brace
            int brace_count = 1;
            size_t obj_end = obj_start + 1;
            while (obj_end < json_str.length() && brace_count > 0) {
                if (json_str[obj_end] == '{') brace_count++;
                else if (json_str[obj_end] == '}') brace_count--;
                obj_end++;
            }
            
            std::string obj_str = json_str.substr(obj_start, obj_end - obj_start);
            
            // Parse dtype
            size_t dtype_pos = obj_str.find("\"dtype\"");
            if (dtype_pos == std::string::npos) {
                pos = obj_end;
                continue;
            }
            
            size_t dtype_quote1 = obj_str.find('"', dtype_pos + 7);
            size_t dtype_quote2 = obj_str.find('"', dtype_quote1 + 1);
            std::string dtype = obj_str.substr(dtype_quote1 + 1, dtype_quote2 - dtype_quote1 - 1);
            
            // Parse shape
            size_t shape_pos = obj_str.find("\"shape\"");
            if (shape_pos == std::string::npos) {
                pos = obj_end;
                continue;
            }
            
            size_t shape_bracket1 = obj_str.find('[', shape_pos);
            size_t shape_bracket2 = obj_str.find(']', shape_bracket1);
            std::string shape_str = obj_str.substr(shape_bracket1 + 1, shape_bracket2 - shape_bracket1 - 1);
            
            std::vector<int64_t> shape;
            std::stringstream shape_ss(shape_str);
            std::string item;
            while (std::getline(shape_ss, item, ',')) {
                // Trim whitespace
                item.erase(0, item.find_first_not_of(" \t"));
                item.erase(item.find_last_not_of(" \t") + 1);
                if (!item.empty()) {
                    shape.push_back(std::stoll(item));
                }
            }
            
            // Parse data_offsets
            size_t offsets_pos = obj_str.find("\"data_offsets\"");
            if (offsets_pos == std::string::npos) {
                pos = obj_end;
                continue;
            }
            
            size_t offsets_bracket1 = obj_str.find('[', offsets_pos);
            size_t offsets_bracket2 = obj_str.find(']', offsets_bracket1);
            std::string offsets_str = obj_str.substr(offsets_bracket1 + 1, offsets_bracket2 - offsets_bracket1 - 1);
            
            std::vector<size_t> offsets;
            std::stringstream offsets_ss(offsets_str);
            while (std::getline(offsets_ss, item, ',')) {
                item.erase(0, item.find_first_not_of(" \t"));
                item.erase(item.find_last_not_of(" \t") + 1);
                if (!item.empty()) {
                    offsets.push_back(std::stoull(item));
                }
            }
            
            if (offsets.size() != 2) {
                throw std::runtime_error("Invalid data_offsets for tensor: " + tensor_name);
            }
            
            SafeTensorInfo tensor_info;
            tensor_info.dtype = dtype;
            tensor_info.shape = shape;
            tensor_info.data_offset = data_section_offset_ + offsets[0];
            tensor_info.data_size = offsets[1] - offsets[0];
            
            tensor_info_[tensor_name] = tensor_info;
            
            pos = obj_end;
        }
    }
};

SafeTensorReader::SafeTensorReader(const std::string& filepath) 
    : impl_(std::make_unique<Impl>(filepath)) {}

SafeTensorReader::~SafeTensorReader() = default;

TensorView SafeTensorReader::get_tensor(const std::string& name) const {
    auto it = impl_->tensor_info_.find(name);
    if (it == impl_->tensor_info_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    const auto& info = it->second;
    DType dtype = parse_dtype(info.dtype);
    
    // Calculate expected size
    size_t expected_size = 1;
    for (int64_t dim : info.shape) {
        expected_size *= static_cast<size_t>(dim);
    }
    expected_size *= dtype_size(dtype);

    if (expected_size != info.data_size) {
        throw std::runtime_error("Tensor size mismatch for: " + name);
    }

    if (info.data_offset + info.data_size > impl_->file_size_) {
        throw std::runtime_error("Tensor data out of bounds: " + name);
    }

    // Create TensorView pointing to memory-mapped data
    const uint8_t* data_ptr = static_cast<const uint8_t*>(impl_->mapped_data_) + info.data_offset;
    void* mutable_ptr = const_cast<void*>(static_cast<const void*>(data_ptr));
    return make_view(mutable_ptr, dtype, info.shape);
}

bool SafeTensorReader::has_tensor(const std::string& name) const {
    return impl_->tensor_info_.find(name) != impl_->tensor_info_.end();
}

const SafeTensorInfo& SafeTensorReader::get_tensor_info(const std::string& name) const {
    auto it = impl_->tensor_info_.find(name);
    if (it == impl_->tensor_info_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

std::vector<std::string> SafeTensorReader::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(impl_->tensor_info_.size());
    for (const auto& [name, _] : impl_->tensor_info_) {
        names.push_back(name);
    }
    return names;
}

} // namespace ie
