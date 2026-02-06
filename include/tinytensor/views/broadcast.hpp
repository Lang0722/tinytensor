#pragma once

#include <cstddef>
#include <vector>

#include "../core/shape.hpp"
#include "../core/strides.hpp"
#include "../utils/assert.hpp"

namespace tt {

template <typename T>
class broadcast_view {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  broadcast_view() = default;

  broadcast_view(T* data, difference_type offset, shape_t broadcast_shape,
                 strides_t strides)
      : data_(data),
        offset_(offset),
        shape_(std::move(broadcast_shape)),
        strides_(std::move(strides)) {}

  [[nodiscard]] size_type ndim() const noexcept { return shape_.ndim(); }
  [[nodiscard]] size_type size() const noexcept { return shape_.size(); }
  [[nodiscard]] const shape_t& shape() const noexcept { return shape_; }
  [[nodiscard]] const strides_t& strides() const noexcept { return strides_; }
  [[nodiscard]] difference_type offset() const noexcept { return offset_; }
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }

  template <typename... Idx>
  [[nodiscard]] const_reference operator()(Idx... indices) const {
    static_assert(sizeof...(Idx) > 0, "At least one index required");
    TT_ASSERT(sizeof...(Idx) == ndim(), "Index count must match dimensions");
    TT_ASSERT(detail::check_bounds(shape_, indices...), "Index out of bounds");
    return data_[offset_ + detail::data_offset(strides_, indices...)];
  }

  class iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    iterator() = default;
    iterator(const broadcast_view* view, std::vector<size_type> indices)
        : view_(view), indices_(std::move(indices)) {}

    reference operator*() const {
      std::ptrdiff_t off = static_cast<std::ptrdiff_t>(view_->offset_);
      for (size_type i = 0; i < indices_.size(); ++i) {
        off += view_->strides_[i] * static_cast<std::ptrdiff_t>(indices_[i]);
      }
      return view_->data_[off];
    }

    iterator& operator++() {
      if (!detail::advance_multi_index(indices_, view_->shape())) {
        indices_.clear();
      }
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
    const broadcast_view* view_ = nullptr;
    std::vector<size_type> indices_;
  };

  [[nodiscard]] iterator begin() const {
    if (size() == 0) return end();
    return iterator(this, std::vector<size_type>(ndim(), 0));
  }

  [[nodiscard]] iterator end() const { return iterator(this, {}); }

 private:
  T* data_ = nullptr;
  difference_type offset_ = 0;
  shape_t shape_;
  strides_t strides_;
};

namespace detail {

inline bool broadcast_shapes(const shape_t& a, const shape_t& b,
                             shape_t& result) {
  std::size_t max_ndim = std::max(a.ndim(), b.ndim());
  result.resize(max_ndim);

  for (std::size_t i = 0; i < max_ndim; ++i) {
    std::size_t a_dim = i < a.ndim() ? a[a.ndim() - 1 - i] : 1;
    std::size_t b_dim = i < b.ndim() ? b[b.ndim() - 1 - i] : 1;

    if (a_dim == b_dim) {
      result[max_ndim - 1 - i] = a_dim;
    } else if (a_dim == 1) {
      result[max_ndim - 1 - i] = b_dim;
    } else if (b_dim == 1) {
      result[max_ndim - 1 - i] = a_dim;
    } else {
      return false;
    }
  }

  return true;
}

inline strides_t broadcast_strides(const shape_t& original_shape,
                                   const strides_t& original_strides,
                                   const shape_t& broadcast_shape) {
  if (broadcast_shape.ndim() < original_shape.ndim()) {
    TT_THROW(broadcast_error, "Cannot broadcast to lower rank");
  }

  strides_t result(broadcast_shape.ndim(), 0);

  std::size_t offset = broadcast_shape.ndim() - original_shape.ndim();

  for (std::size_t i = 0; i < original_shape.ndim(); ++i) {
    if (original_shape[i] == broadcast_shape[offset + i]) {
      result[offset + i] = original_strides[i];
    } else if (original_shape[i] == 1) {
      result[offset + i] = 0;
    } else {
      TT_THROW(broadcast_error, "Cannot broadcast shapes");
    }
  }

  return result;
}

}  // namespace detail

template <typename T>
broadcast_view<T> broadcast(T* data, std::ptrdiff_t offset,
                            const shape_t& original_shape,
                            const strides_t& original_strides,
                            const shape_t& target_shape) {
  auto new_strides =
      detail::broadcast_strides(original_shape, original_strides, target_shape);
  return broadcast_view<T>(data, offset, target_shape, std::move(new_strides));
}

}  // namespace tt
