#pragma once 
#include "infer_engine/core/types.hpp"
#include <memory> 
#include <vector>
#include <cstring> 

namespace ie{ 

    inline std::vector<int64_t> row_major_strides(const std::vector<int64_t>& shape){

        //strides in *elements*
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1; 
    for (int64_t i = (int64_t)shape.size()-1; i >=0; --i){
        s[i] = acc; 
        acc *= shape[i];
    }
    return s; 
    }


    struct TensorView{

        void* data = nullptr; 
        DType dt = DType::F32; 
        std::vector<int64_t> shape;
        std::vector<int64_t> stride; 

        bool defined() const { return data != nullptr; }
        int64_t rank() const { return (int64_t)shape.size();}
        int64_t numel() const {
            int64_t n=1; for (auto d: shape) n*=d; return n; 
        }
        size_t itemsize() const { return dtype_bytes(dt); }
        size_t nbytes() const {return (size_t)numel() * itemsize();}
        bool is_contiguous()  const{
           return stride == row_major_strides(shape);
        }  

        template <typename T> T* ptr() { return reinterpret_cast<T*>(data);}
        template <typename T> const T* ptr() const {return reinterpret_cast<const T*>(data);}


    };

    struct Tensor { 
        std::unique_ptr<uint8_t[]> storage; 
        TensorView view; 

        static Tensor empty(const std::vector<int64_t>& shape, DType dt){
            Tensor t; 
            size_t bytes = (size_t)ie::Shape{shape}.numel()*dtype_bytes(dt);
            t.storage = std::unique_ptr<uint8_t[]>(new uint8_t[bytes]());
            t.view.data = t.storage.get();
            t.view.dt = dt;
            t.view.shape = shape;
            t.view.stride = row_major_strides(shape);
            return t;
        }

        static Tensor from_raw(const void* src, const std::vector<int64_t>& shape, DType dt){

            Tensor t = empty(shape, dt);
            std::memcpy(t.view.data, src, t.view.nbytes());
            return t; 
        }
    }; 

    Tensor astype_copy(const TensorView& src, DType dst);

    inline TensorView make_view(void* data, DType dt, const std::vector<int64_t>& shape, const std::vector<int64_t>& stride = {}){
      
        TensorView v; 
        v.data = data; v.dt = dt; v.shape = shape; 
        v.stride = stride.empty() ? row_major_strides(shape) : stride;
        return v;

    }

} //namespace ie