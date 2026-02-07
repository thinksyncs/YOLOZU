#include <cuda_runtime_api.h>

#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bin_io.h"
#include "shape.h"
#include "yolozu_predictions.h"

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[trt] " << msg << "\n";
    }
  }
};

static void print_usage() {
  std::cerr << "yolozu_infer_trt \\\n";
  std::cerr << "  --engine <model.plan> \\\n";
  std::cerr << "  --input <input.f32.bin> --input-shape 1x3x640x640 --input-binding images \\\n";
  std::cerr << "  --combined-binding output0 --boxes-scale abs --input-size 640x640 \\\n";
  std::cerr << "  --image <image_path_for_meta> --output <predictions.json>\n";
}

static std::pair<int, int> parse_size(const std::string& spec) {
  const auto pos = spec.find('x');
  if (pos == std::string::npos) {
    throw std::runtime_error("input-size must be 'WxH'");
  }
  const int w = std::stoi(spec.substr(0, pos));
  const int h = std::stoi(spec.substr(pos + 1));
  if (w <= 0 || h <= 0) {
    throw std::runtime_error("input-size dims must be positive");
  }
  return {w, h};
}

static std::vector<char> read_all(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open engine: " + path);
  }
  file.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  if (size > 0) {
    file.read(data.data(), static_cast<std::streamsize>(size));
  }
  if (!file) {
    throw std::runtime_error("failed to read engine: " + path);
  }
  return data;
}

static void cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error (") + what + "): " + cudaGetErrorString(err));
  }
}

int main(int argc, char** argv) {
  std::string engine_path;
  std::string input_path;
  std::string input_shape_spec;
  std::string input_binding = "images";
  std::string combined_binding = "output0";
  std::string boxes_scale = "abs";
  std::string input_size_spec;
  std::string image_path;
  std::string output_path;
  float min_score = 0.0f;
  int topk = 0;

  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    }
    auto take = [&](std::string& dst) {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + arg);
      }
      dst = argv[++i];
    };

    if (arg == "--engine") {
      take(engine_path);
    } else if (arg == "--input") {
      take(input_path);
    } else if (arg == "--input-shape") {
      take(input_shape_spec);
    } else if (arg == "--input-binding") {
      take(input_binding);
    } else if (arg == "--combined-binding") {
      take(combined_binding);
    } else if (arg == "--boxes-scale") {
      take(boxes_scale);
    } else if (arg == "--input-size") {
      take(input_size_spec);
    } else if (arg == "--image") {
      take(image_path);
    } else if (arg == "--output") {
      take(output_path);
    } else if (arg == "--min-score") {
      std::string v;
      take(v);
      min_score = std::stof(v);
    } else if (arg == "--topk") {
      std::string v;
      take(v);
      topk = std::stoi(v);
    } else {
      std::cerr << "unknown arg: " << arg << "\n";
      print_usage();
      return 2;
    }
  }

  if (engine_path.empty() || input_path.empty() || input_shape_spec.empty() || output_path.empty() ||
      image_path.empty() || input_size_spec.empty()) {
    print_usage();
    return 2;
  }

  const auto input_shape = yolozu_parse_shape(input_shape_spec);
  const auto expected = yolozu_numel(input_shape);
  const auto input_data = yolozu_read_f32(input_path, expected);
  const auto [input_w, input_h] = parse_size(input_size_spec);

  TrtLogger logger;
  auto engine_blob = read_all(engine_path);

  std::unique_ptr<nvinfer1::IRuntime, void (*)(nvinfer1::IRuntime*)> runtime(
      nvinfer1::createInferRuntime(logger),
      [](nvinfer1::IRuntime* p) { if (p) p->destroy(); });
  if (!runtime) {
    throw std::runtime_error("failed to create TensorRT runtime");
  }

  std::unique_ptr<nvinfer1::ICudaEngine, void (*)(nvinfer1::ICudaEngine*)> engine(
      runtime->deserializeCudaEngine(engine_blob.data(), engine_blob.size()),
      [](nvinfer1::ICudaEngine* p) { if (p) p->destroy(); });
  if (!engine) {
    throw std::runtime_error("failed to deserialize engine: " + engine_path);
  }

  std::unique_ptr<nvinfer1::IExecutionContext, void (*)(nvinfer1::IExecutionContext*)> context(
      engine->createExecutionContext(),
      [](nvinfer1::IExecutionContext* p) { if (p) p->destroy(); });
  if (!context) {
    throw std::runtime_error("failed to create execution context");
  }

  const int input_index = engine->getBindingIndex(input_binding.c_str());
  const int output_index = engine->getBindingIndex(combined_binding.c_str());
  if (input_index < 0 || output_index < 0) {
    throw std::runtime_error("binding not found (check --input-binding / --combined-binding)");
  }

  // This template assumes static shapes. For dynamic shapes, extend this by calling setBindingDimensions.
  const auto out_dims = engine->getBindingDimensions(output_index);
  size_t out_count = 1;
  for (int i = 0; i < out_dims.nbDims; i++) {
    out_count *= static_cast<size_t>(out_dims.d[i]);
  }
  if (out_count == 0) {
    throw std::runtime_error("output element count is zero (dynamic shapes not supported in this template)");
  }

  void* d_input = nullptr;
  void* d_output = nullptr;
  cuda_check(cudaMalloc(&d_input, input_data.size() * sizeof(float)), "cudaMalloc(input)");
  cuda_check(cudaMalloc(&d_output, out_count * sizeof(float)), "cudaMalloc(output)");

  cuda_check(cudaMemcpy(d_input, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(input)");

  std::vector<void*> bindings(engine->getNbBindings(), nullptr);
  bindings[static_cast<size_t>(input_index)] = d_input;
  bindings[static_cast<size_t>(output_index)] = d_output;

  const bool ok = context->executeV2(bindings.data());
  if (!ok) {
    throw std::runtime_error("TensorRT executeV2 failed");
  }

  std::vector<float> out_flat(out_count);
  cuda_check(cudaMemcpy(out_flat.data(), d_output, out_count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(output)");

  cuda_check(cudaFree(d_input), "cudaFree(input)");
  cuda_check(cudaFree(d_output), "cudaFree(output)");

  auto dets = yolozu_from_combined_xyxy_score_class(
      out_flat,
      input_w,
      input_h,
      boxes_scale,
      min_score,
      topk);

  yolozu_write_predictions_json(output_path, image_path, dets, "tensorrt", engine_path);
  std::cout << output_path << "\n";
  return 0;
}

