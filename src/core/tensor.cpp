#include "infer_engine/core/tensor.hpp"
#include <cmath>
#include <cstring>

namespace ie {

// naive converters (good enough for Day 1 tests)
static inline float f16_to_f32(uint16_t h) {
  // IEEE 754 half → float (fast-ish, not bit-perfect; fine for loader sanity)
  uint16_t h_exp = (h & 0x7C00u) >> 10;
  uint16_t h_sig = (h & 0x03FFu);
  uint32_t sign = (h & 0x8000u) ? 0x80000000u : 0u;
  if (h_exp == 0) {
    // subnorm/zero
    if (h_sig == 0) return std::bit_cast<float>(sign);
    float f = (float)h_sig / 1024.0f;
    int e = -14;
    return std::ldexp(f, e);
  } else if (h_exp == 0x1F) {
    // inf/nan
    uint32_t bits = sign | 0x7F800000u | ((uint32_t)h_sig << 13);
    return std::bit_cast<float>(bits);
  } else {
    int e = (int)h_exp - 15;
    float f = 1.0f + (float)h_sig / 1024.0f;
    return std::ldexp(f, e);
  }
}


static inline float bf16_to_f32(uint16_t b) {
  uint32_t bits = ((uint32_t)b) << 16;\
  
  return std::bit_cast<float>(bits);
}



Tensor astype_copy(const TensorView& src, DType dst) {
  IE_CHECK(src.is_contiguous(), "astype_copy requires contiguous src");
  auto out = Tensor::empty(src.shape, dst);

  if (src.dt == dst) {
    std::memcpy(out.view.data, src.data, src.nbytes());
    return out;
  }

  const int64_t N = src.numel();
  if (dst == DType::F32) {
    float* o = out.view.ptr<float>();
    if (src.dt == DType::F16) {
      const uint16_t* p = src.ptr<const uint16_t>();
      for (int64_t i = 0; i < N; ++i) o[i] = f16_to_f32(p[i]);
    } else if (src.dt == DType::BF16) {
      const uint16_t* p = src.ptr<const uint16_t>();
      for (int64_t i = 0; i < N; ++i) o[i] = bf16_to_f32(p[i]);
    } else if (src.dt == DType::I8) {
      const int8_t* p = src.ptr<const int8_t>();
      for (int64_t i = 0; i < N; ++i) o[i] = (float)p[i];
    } else {
      IE_CHECK(false, "unsupported src->f32");
    }
    return out;
  }

  IE_CHECK(false, "Only conversions to f32 implemented in Day1");
  return out;
}

} // namespace ie