#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<float> yolozu_read_f32(const std::string& path, size_t expected_count) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open input: " + path);
  }
  file.seekg(0, std::ios::end);
  const auto size_bytes = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  if (size_bytes % sizeof(float) != 0) {
    throw std::runtime_error("input size is not a multiple of float32: " + path);
  }
  const auto count = size_bytes / sizeof(float);
  if (expected_count != 0 && count != expected_count) {
    throw std::runtime_error("input float count mismatch (expected " + std::to_string(expected_count) +
                             ", got " + std::to_string(count) + "): " + path);
  }

  std::vector<float> data(count);
  if (!data.empty()) {
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size_bytes));
    if (!file) {
      throw std::runtime_error("failed to read input: " + path);
    }
  }
  return data;
}

inline void yolozu_write_f32(const std::string& path, const float* data, size_t count) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open output: " + path);
  }
  if (count > 0) {
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
    if (!file) {
      throw std::runtime_error("failed to write output: " + path);
    }
  }
}

