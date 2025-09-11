#include "infer_engine/layers/ops/reshape.hpp"
#include <stdexcept>

namespace ie {
namespace ops {

Tensor reshape(const Tensor& x, const std::vector<int64_t>& new_shape) {
    // Validate product(new_shape) == x.view.numel()
    int64_t product = 1; 

    for(size_t i = 0; i < new_shape.size(); i++){
        product *= new_shape[i];
    }
    
    if (product != x.view.numel()){
        throw std::invalid_argument("reshape: new shape has different number of elements");
    }

    // Create new tensor with same storage but new shape
    Tensor result;
    result.storage = std::unique_ptr<uint8_t[]>(new uint8_t[x.view.nbytes()]);
    std::memcpy(result.storage.get(), x.view.data, x.view.nbytes());
    
    result.view.data = result.storage.get();
    result.view.dt = x.view.dt;
    result.view.shape = new_shape;
    result.view.stride = row_major_strides(new_shape);
    
    return result;
}

} // namespace ops
} // namespace ie
