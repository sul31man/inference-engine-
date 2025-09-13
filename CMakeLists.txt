cmake_minimum_required(VERSION 3.16)
project(inference_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optimize build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# Core library
file(GLOB_RECURSE IE_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/graph/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/io/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/layers/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/layers/ops/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp
)

add_library(infer_engine ${IE_SRC})
target_include_directories(infer_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link Accelerate on macOS
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  if(ACCELERATE_FRAMEWORK)
    target_link_libraries(infer_engine PUBLIC ${ACCELERATE_FRAMEWORK})
  endif()
endif()

# OpenMP (CPU parallelism)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(infer_engine PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(infer_engine PUBLIC IE_OMP=1)
endif()

# Examples
add_executable(run_greedy_cpu examples/run_greedy_cpu.cpp)
target_link_libraries(run_greedy_cpu PRIVATE infer_engine)

add_executable(load_mistral examples/load_mistral.cpp)
target_link_libraries(load_mistral PRIVATE infer_engine)

add_executable(run_mistral_inference examples/run_mistral_inference.cpp)
target_link_libraries(run_mistral_inference PRIVATE infer_engine)

add_executable(debug_mistral examples/debug_mistral.cpp)
target_link_libraries(debug_mistral PRIVATE infer_engine)

add_executable(inspect_safetensors examples/inspect_safetensors.cpp)
target_link_libraries(inspect_safetensors PRIVATE infer_engine)

add_executable(test_gqa_minimal examples/test_gqa_minimal.cpp)
target_link_libraries(test_gqa_minimal PRIVATE infer_engine)

add_executable(fast_logits examples/fast_logits.cpp)
target_link_libraries(fast_logits PRIVATE infer_engine)

add_executable(text_gen examples/text_gen.cpp)
target_link_libraries(text_gen PRIVATE infer_engine)

# Unit tests (simple executables with mains inside)
add_executable(test_kv_cache tests/unit/test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE infer_engine)

add_executable(test_attention_mlp tests/unit/test_attention_mlp.cpp)
target_link_libraries(test_attention_mlp PRIVATE infer_engine)

add_executable(test_runtime_ctx tests/unit/test_runtime_ctx.cpp)
target_link_libraries(test_runtime_ctx PRIVATE infer_engine)


