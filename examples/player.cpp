#pragma once 
#include <cstdint> 
#include <vector> 
#include <stdexcept> 
#include <cstring> 





// Given q[d_h] F32, K/V cached in F16/BF16 over positions [0..pos]
void attend_one_head(const float* q, const uint8_t* Kbase, const uint8_t* Vbase,
    int64_t pos, int64_t n_kv, int64_t kvh, int64_t d_h, DType dt,
    float* out)
{
std::vector<float> scores(pos+1,0.f);
size_t elem=dtype_size(dt), perpos = (size_t)n_kv*d_h*elem;
for (int64_t p=0;p<=pos;++p){
const uint8_t* kpos=Kbase + p*perpos + kvh*d_h*elem;
float dot=0.f;
for (int64_t j=0;j<d_h;++j){
float kj = (dt==DType::F16)? f16_to_f32(((uint16_t*)kpos)[j])
                      : bf16_to_f32(((uint16_t*)kpos)[j]);
dot += q[j]*kj;
}
scores[p] = dot/std::sqrt((float)d_h);
}
float m=-1e30f; for(float s: scores) m = s>m? s:m;
double sum=0.0; for(float& s: scores){ s=std::exp(s-m); sum+=s; }
float inv=(float)(1.0/sum);
for(float& s: scores) s*=inv;

std::fill(out, out+d_h, 0.f);
for (int64_t p=0;p<=pos;++p){
const uint8_t* vpos=Vbase + p*perpos + kvh*d_h*elem;
for (int64_t j=0;j<d_h;++j){
float vj = (dt==DType::F16)? f16_to_f32(((uint16_t*)vpos)[j])
                      : bf16_to_f32(((uint16_t*)vpos)[j]);
out[j] += scores[p]*vj;
}
}
}


// Given q[d_h] F32, K/V cached in F16/BF16 over positions [0..pos]
void attend_one_head(const float* q, const uint8_t* Kbase, const uint8_t* Vbase,
    int64_t pos, int64_t n_kv, int64_t kvh, int64_t d_h, DType dt,
    float* out)
{
std::vector<float> scores(pos+1,0.f);
size_t elem=dtype_size(dt), perpos = (size_t)n_kv*d_h*elem;
for (int64_t p=0;p<=pos;++p){
const uint8_t* kpos=Kbase + p*perpos + kvh*d_h*elem;
float dot=0.f;
for (int64_t j=0;j<d_h;++j){
float kj = (dt==DType::F16)? f16_to_f32(((uint16_t*)kpos)[j])
                      : bf16_to_f32(((uint16_t*)kpos)[j]);
dot += q[j]*kj;
}
scores[p] = dot/std::sqrt((float)d_h);
}
float m=-1e30f; for(float s: scores) m = s>m? s:m;
double sum=0.0; for(float& s: scores){ s=std::exp(s-m); sum+=s; }
float inv=(float)(1.0/sum);
for(float& s: scores) s*=inv;

std::fill(out, out+d_h, 0.f);
for (int64_t p=0;p<=pos;++p){
const uint8_t* vpos=Vbase + p*perpos + kvh*d_h*elem;
for (int64_t j=0;j<d_h;++j){
float vj = (dt==DType::F16)? f16_to_f32(((uint16_t*)vpos)[j])
                      : bf16_to_f32(((uint16_t*)vpos)[j]);
out[j] += scores[p]*vj;
}
}
}