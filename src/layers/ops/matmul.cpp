#include "infer_engine/layers/ops/matmul.hpp"
#include <stdexcept>
#include "infer_engine/core/tensor2.hpp"

namespace ie {
namespace ops {

Tensor matmul(const TensorView& A, const TensorView& B, bool transpose_b) {
    // TODO: implement naive matmul for 2D tensors, support transpose of B
    throw std::logic_error("matmul not implemented");
}

auto A_shape = A.shape; 
auto B_shape = B.shape; 

int x = A_shape[0];
int y = A_shape[1];
int y_2 = B_shape[0];
int z = B_shape[1];

if(y != y_2){
     
    std::logic_error("dimensions are not compatible for matrix multiplication");

}


Tensor output; 

output.view.shape = {x, z};

output.storage = std::unique_ptr<uint8_t[]>(new uint8_t[A.view.nbytes()*B.view.nbytes() / (y**2)]);
output.veiw.dt = A.view.dt; 
output.view.stride = row_major_strides(new_shape);


for (int i=0; i < x; i++){

    for (int j = 0; j < z; j++){
        
        int64_t k = 0; 

        for (int k = 0; k < y; k++ ){


        }
    }
}

 
} // namespace ops
} // namespace ie


