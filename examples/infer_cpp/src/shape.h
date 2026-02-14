#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<int64_t> yolozu_parse_shape(const std::string& spec) {
  if (spec.empty()) {
    throw std::runtime_error("shape spec must be non-empty");
  }
  std::vector<int64_t> dims;
  std::string token;
  std::istringstream stream(spec);
  while (std::getline(stream, token, 'x')) {
    if (token.empty()) {
      throw std::runtime_error("invalid shape spec: empty dim");
    }
    try {
      const auto value = std::stoll(token);
      if (value <= 0) {
        throw std::runtime_error("invalid shape spec: dims must be positive");
      }
      dims.push_back(static_cast<int64_t>(value));
    } catch (const std::exception&) {
      throw std::runtime_error("invalid shape spec: expected '1x3x640x640'");
    }
  }
  if (dims.empty()) {
    throw std::runtime_error("invalid shape spec: no dims");
  }
  return dims;
}

inline size_t yolozu_numel(const std::vector<int64_t>& dims) {
  size_t total = 1;
  for (const auto dim : dims) {
    if (dim <= 0) {
      throw std::runtime_error("numel: dims must be positive");
    }
    total *= static_cast<size_t>(dim);
  }
  return total;
}

