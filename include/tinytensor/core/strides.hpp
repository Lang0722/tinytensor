#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "layout.hpp"
#include "shape.hpp"

namespace tt {

using strides_t = std::vector<std::ptrdiff_t>;

namespace detail {

inline strides_t compute_strides(const shape_t& shape,
                                 layout_type layout = default_layout) {
  if (shape.empty()) return {};

  strides_t strides(shape.ndim());
  std::ptrdiff_t data_size = 1;

  if (layout == layout_type::row_major) {
    for (std::size_t i = shape.ndim(); i != 0; --i) {
      strides[i - 1] = (shape[i - 1] == 1) ? 0 : data_size;
      data_size *= static_cast<std::ptrdiff_t>(shape[i - 1]);
    }
  } else {
    for (std::size_t i = 0; i < shape.ndim(); ++i) {
      strides[i] = (shape[i] == 1) ? 0 : data_size;
      data_size *= static_cast<std::ptrdiff_t>(shape[i]);
    }
  }

  return strides;
}

template <typename... Idx>
constexpr std::ptrdiff_t data_offset(const strides_t& strides,
                                     Idx... indices) noexcept {
  std::ptrdiff_t offset = 0;
  std::size_t i = 0;
  ((offset += strides[i++] * static_cast<std::ptrdiff_t>(indices)), ...);
  return offset;
}

inline std::ptrdiff_t data_offset(
    const strides_t& strides, std::span<const std::size_t> indices) noexcept {
  std::ptrdiff_t offset = 0;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    offset += strides[i] * static_cast<std::ptrdiff_t>(indices[i]);
  }
  return offset;
}

template <typename... Idx>
constexpr bool check_bounds(const shape_t& shape, Idx... indices) noexcept {
  std::size_t i = 0;
  bool in_bounds = true;
  ((in_bounds = in_bounds && (static_cast<std::size_t>(indices) < shape[i++])),
   ...);
  return in_bounds;
}

inline bool check_bounds(const shape_t& shape,
                         std::span<const std::size_t> indices) noexcept {
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape[i]) return false;
  }
  return true;
}

}  // namespace detail

}  // namespace tt
