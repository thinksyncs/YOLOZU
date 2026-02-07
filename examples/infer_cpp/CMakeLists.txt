cmake_minimum_required(VERSION 3.20)
project(yolozu_infer_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(yolozu_infer_stub
  src/main_stub.cpp
)

target_include_directories(yolozu_infer_stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

option(YOLOZU_INFER_BUILD_ONNXRT "Build ONNXRuntime runner when deps are found" ON)
option(YOLOZU_INFER_BUILD_TRT "Build TensorRT runner when deps are found" ON)

# -------------------------
# ONNXRuntime (optional)
# -------------------------
if(YOLOZU_INFER_BUILD_ONNXRT)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNXRuntime C/C++ package root (contains include/ and lib/)")
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
      ${ONNXRUNTIME_ROOT}
      ${ONNXRUNTIME_ROOT}/include
      $ENV{ONNXRUNTIME_ROOT}
      $ENV{ONNXRUNTIME_ROOT}/include
  )
  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
      ${ONNXRUNTIME_ROOT}
      ${ONNXRUNTIME_ROOT}/lib
      $ENV{ONNXRUNTIME_ROOT}
      $ENV{ONNXRUNTIME_ROOT}/lib
  )
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    add_executable(yolozu_infer_onnxrt
      src/main_onnxrt.cpp
    )
    target_include_directories(yolozu_infer_onnxrt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(yolozu_infer_onnxrt PRIVATE ${ONNXRUNTIME_LIBRARY})
  else()
    message(STATUS "ONNXRuntime not found; skipping yolozu_infer_onnxrt (set -DONNXRUNTIME_ROOT=... to enable)")
  endif()
endif()

# -------------------------
# TensorRT (optional)
# -------------------------
if(YOLOZU_INFER_BUILD_TRT)
  find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h)
  find_library(TENSORRT_NVINFER_LIBRARY NAMES nvinfer)
  find_library(TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin)
  find_library(CUDA_CUDART_LIBRARY NAMES cudart)

  if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIBRARY AND CUDA_CUDART_LIBRARY)
    add_executable(yolozu_infer_trt
      src/main_trt.cpp
    )
    target_include_directories(yolozu_infer_trt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(yolozu_infer_trt PRIVATE ${TENSORRT_NVINFER_LIBRARY} ${CUDA_CUDART_LIBRARY})
    if(TENSORRT_PLUGIN_LIBRARY)
      target_link_libraries(yolozu_infer_trt PRIVATE ${TENSORRT_PLUGIN_LIBRARY})
    endif()
  else()
    message(STATUS "TensorRT not found; skipping yolozu_infer_trt (build inside an NVIDIA/TensorRT environment)")
  endif()
endif()

