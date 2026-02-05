#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "../core/shape.hpp"
#include "../core/strides.hpp"
#include "../utils/assert.hpp"
#include "slice.hpp"

namespace tt {

template <typename T>
class tensor_view {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  tensor_view() = default;

  tensor_view(T* data, difference_type offset, shape_t shape, strides_t strides)
      : data_(data),
        offset_(offset),
        shape_(std::move(shape)),
        strides_(std::move(strides)) {}

  [[nodiscard]] size_type ndim() const noexcept { return shape_.ndim(); }
  [[nodiscard]] size_type size() const noexcept { return shape_.size(); }
  [[nodiscard]] const shape_t& shape() const noexcept { return shape_; }
  [[nodiscard]] const strides_t& strides() const noexcept { return strides_; }
  [[nodiscard]] difference_type offset() const noexcept { return offset_; }
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }

  template <typename... Idx>
  [[nodiscard]] reference operator()(Idx... indices) {
    static_assert(sizeof...(Idx) > 0, "At least one index required");
    TT_ASSERT(sizeof...(Idx) == ndim(), "Index count must match dimensions");
    TT_ASSERT(detail::check_bounds(shape_, indices...), "Index out of bounds");
    return data_[offset_ + detail::data_offset(strides_, indices...)];
  }

  template <typename... Idx>
  [[nodiscard]] const_reference operator()(Idx... indices) const {
    static_assert(sizeof...(Idx) > 0, "At least one index required");
    TT_ASSERT(sizeof...(Idx) == ndim(), "Index count must match dimensions");
    TT_ASSERT(detail::check_bounds(shape_, indices...), "Index out of bounds");
    return data_[offset_ + detail::data_offset(strides_, indices...)];
  }

  [[nodiscard]] reference at(std::span<const size_type> indices) {
    TT_ASSERT(indices.size() == ndim(), "Index count must match dimensions");
    TT_ASSERT(detail::check_bounds(shape_, indices), "Index out of bounds");
    return data_[offset_ + detail::data_offset(strides_, indices)];
  }

  [[nodiscard]] const_reference at(std::span<const size_type> indices) const {
    TT_ASSERT(indices.size() == ndim(), "Index count must match dimensions");
    TT_ASSERT(detail::check_bounds(shape_, indices), "Index out of bounds");
    return data_[offset_ + detail::data_offset(strides_, indices)];
  }

  [[nodiscard]] bool is_contiguous() const noexcept {
    if (shape_.empty()) return true;

    // Check row-major contiguity
    std::ptrdiff_t expected = 1;
    bool row_major = true;
    for (size_type i = ndim(); i != 0; --i) {
      if (shape_[i - 1] == 1) continue;
      if (strides_[i - 1] != expected) { row_major = false; break; }
      expected *= static_cast<std::ptrdiff_t>(shape_[i - 1]);
    }
    if (row_major) return true;

    // Check column-major contiguity
    expected = 1;
    for (size_type i = 0; i < ndim(); ++i) {
      if (shape_[i] == 1) continue;
      if (strides_[i] != expected) return false;
      expected *= static_cast<std::ptrdiff_t>(shape_[i]);
    }
    return true;
  }

  class iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    iterator() = default;
    iterator(tensor_view* view, std::vector<size_type> indices)
        : view_(view), indices_(std::move(indices)) {}

    reference operator*() const { return view_->at(indices_); }
    pointer operator->() const { return &view_->at(indices_); }

    iterator& operator++() {
      for (size_type i = view_->ndim(); i != 0; --i) {
        if (++indices_[i - 1] < view_->shape()[i - 1]) {
          return *this;
        }
        indices_[i - 1] = 0;
      }
      indices_.clear();
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const iterator& other) const {
      if (indices_.empty() && other.indices_.empty()) return true;
      return view_ == other.view_ && indices_ == other.indices_;
    }

   private:
    tensor_view* view_ = nullptr;
    std::vector<size_type> indices_;
  };

  class const_iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    const_iterator() = default;
    const_iterator(const tensor_view* view, std::vector<size_type> indices)
        : view_(view), indices_(std::move(indices)) {}

    reference operator*() const { return view_->at(indices_); }
    pointer operator->() const { return &view_->at(indices_); }

    const_iterator& operator++() {
      for (size_type i = view_->ndim(); i != 0; --i) {
        if (++indices_[i - 1] < view_->shape()[i - 1]) {
          return *this;
        }
        indices_[i - 1] = 0;
      }
      indices_.clear();
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const const_iterator& other) const {
      if (indices_.empty() && other.indices_.empty()) return true;
      return view_ == other.view_ && indices_ == other.indices_;
    }

   private:
    const tensor_view* view_ = nullptr;
    std::vector<size_type> indices_;
  };

  [[nodiscard]] iterator begin() {
    if (size() == 0) return end();
    return iterator(this, std::vector<size_type>(ndim(), 0));
  }

  [[nodiscard]] iterator end() { return iterator(this, {}); }

  [[nodiscard]] const_iterator begin() const {
    if (size() == 0) return end();
    return const_iterator(this, std::vector<size_type>(ndim(), 0));
  }

  [[nodiscard]] const_iterator end() const { return const_iterator(this, {}); }

 private:
  T* data_ = nullptr;
  difference_type offset_ = 0;
  shape_t shape_;
  strides_t strides_;
};

namespace detail {

template <typename T>
tensor_view<T> apply_slices(T* data, std::ptrdiff_t base_offset,
                            const shape_t& shape, const strides_t& strides,
                            std::span<const slice_t> slices) {
  using size_type = std::size_t;

  shape_t new_shape;
  strides_t new_strides;
  std::ptrdiff_t new_offset = base_offset;

  size_type dim_idx = 0;
  size_type slice_idx = 0;
  bool seen_ellipsis = false;

  // Count non-newaxis slices to validate against dimensions
  size_type non_newaxis_count = 0;
  size_type ellipsis_count = 0;
  for (const auto& s : slices) {
    if (std::holds_alternative<ellipsis_tag>(s)) {
      ++ellipsis_count;
    } else if (!std::holds_alternative<newaxis_tag>(s)) {
      ++non_newaxis_count;
    }
  }
  TT_ASSERT(ellipsis_count <= 1, "Only one ellipsis allowed in slice");
  TT_ASSERT(non_newaxis_count <= shape.ndim() || ellipsis_count > 0,
            "Too many indices for tensor");

  while (slice_idx < slices.size()) {
    const auto& s = slices[slice_idx];

    if (std::holds_alternative<ellipsis_tag>(s)) {
      TT_ASSERT(!seen_ellipsis, "Only one ellipsis allowed in slice");
      seen_ellipsis = true;

      size_type remaining_slices = slices.size() - slice_idx - 1;
      size_type remaining_dims = shape.ndim() - dim_idx;

      // Count non-newaxis slices remaining
      size_type remaining_non_newaxis = 0;
      for (size_type i = slice_idx + 1; i < slices.size(); ++i) {
        if (!std::holds_alternative<newaxis_tag>(slices[i]) &&
            !std::holds_alternative<ellipsis_tag>(slices[i])) {
          ++remaining_non_newaxis;
        }
      }

      // dims_to_consume is how many dimensions the ellipsis covers
      size_type dims_to_consume = (remaining_dims > remaining_non_newaxis)
                                      ? remaining_dims - remaining_non_newaxis
                                      : 0;

      for (size_type i = 0; i < dims_to_consume; ++i) {
        new_shape.push_back(shape[dim_idx]);
        new_strides.push_back(strides[dim_idx]);
        ++dim_idx;
      }
      ++slice_idx;
      continue;
    }

    if (std::holds_alternative<newaxis_tag>(s)) {
      new_shape.push_back(1);
      new_strides.push_back(0);
      ++slice_idx;
      continue;
    }

    TT_ASSERT(dim_idx < shape.ndim(), "Too many indices for tensor");

    if (std::holds_alternative<std::ptrdiff_t>(s)) {
      std::ptrdiff_t idx = std::get<std::ptrdiff_t>(s);
      idx = normalize_index(idx, shape[dim_idx]);
      TT_ASSERT(idx >= 0 && static_cast<size_type>(idx) < shape[dim_idx],
                "Index out of bounds");
      new_offset += strides[dim_idx] * idx;
      ++dim_idx;
    } else if (std::holds_alternative<all_tag>(s)) {
      new_shape.push_back(shape[dim_idx]);
      new_strides.push_back(strides[dim_idx]);
      ++dim_idx;
    } else if (std::holds_alternative<range_t>(s)) {
      auto nr = normalize_range(std::get<range_t>(s), shape[dim_idx]);
      // For range slicing, offset to the start position
      new_offset += strides[dim_idx] * static_cast<std::ptrdiff_t>(nr.start);
      new_shape.push_back(nr.size);
      new_strides.push_back(strides[dim_idx] * nr.step);
      ++dim_idx;
    }

    ++slice_idx;
  }

  while (dim_idx < shape.ndim()) {
    new_shape.push_back(shape[dim_idx]);
    new_strides.push_back(strides[dim_idx]);
    ++dim_idx;
  }

  return tensor_view<T>(data, new_offset, std::move(new_shape),
                        std::move(new_strides));
}

}  // namespace detail

}  // namespace tt
