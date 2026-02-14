#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "bin_io.h"
#include "shape.h"
#include "yolozu_predictions.h"

static void print_usage() {
  std::cerr << "yolozu_infer_onnxrt \\\n";
  std::cerr << "  --onnx <model.onnx> \\\n";
  std::cerr << "  --input <input.f32.bin> --input-shape 1x3x640x640 --input-name images \\\n";
  std::cerr << "  --combined-output output0 --boxes-scale abs --input-size 640x640 \\\n";
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

int main(int argc, char** argv) {
  std::string onnx_path;
  std::string input_path;
  std::string input_shape_spec;
  std::string input_name = "images";
  std::string combined_output = "output0";
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

    if (arg == "--onnx") {
      take(onnx_path);
    } else if (arg == "--input") {
      take(input_path);
    } else if (arg == "--input-shape") {
      take(input_shape_spec);
    } else if (arg == "--input-name") {
      take(input_name);
    } else if (arg == "--combined-output") {
      take(combined_output);
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

  if (onnx_path.empty() || input_path.empty() || input_shape_spec.empty() || output_path.empty() ||
      image_path.empty() || input_size_spec.empty()) {
    print_usage();
    return 2;
  }

  const auto input_shape = yolozu_parse_shape(input_shape_spec);
  const auto expected = yolozu_numel(input_shape);
  const auto input_data = yolozu_read_f32(input_path, expected);
  const auto [input_w, input_h] = parse_size(input_size_spec);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolozu_infer_onnxrt");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  Ort::Session session(env, onnx_path.c_str(), session_options);

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      const_cast<float*>(input_data.data()),
      input_data.size(),
      input_shape.data(),
      input_shape.size());

  const char* input_names[] = {input_name.c_str()};
  const char* output_names[] = {combined_output.c_str()};

  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
  if (outputs.size() != 1) {
    throw std::runtime_error("unexpected output count");
  }

  const auto& out = outputs[0];
  if (!out.IsTensor()) {
    throw std::runtime_error("output is not a tensor");
  }

  auto info = out.GetTensorTypeAndShapeInfo();
  const auto out_shape = info.GetShape();
  const auto out_count = info.GetElementCount();
  const float* out_data = out.GetTensorData<float>();

  std::vector<float> flat(out_data, out_data + out_count);

  auto dets = yolozu_from_combined_xyxy_score_class(
      flat,
      input_w,
      input_h,
      boxes_scale,
      min_score,
      topk);

  yolozu_write_predictions_json(output_path, image_path, dets, "onnxruntime", onnx_path);
  std::cout << output_path << "\n";
  return 0;
}

