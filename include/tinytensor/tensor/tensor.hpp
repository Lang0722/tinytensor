#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

#include "../core/layout.hpp"
#include "../core/shape.hpp"
#include "../core/strides.hpp"
#include "../utils/assert.hpp"
#include "../views/broadcast.hpp"
#include "../views/slice.hpp"
#include "../views/tensor_view.hpp"

namespace tt {

template <typename T>
class tensor {
  // Prevent tensor<bool> - std::vector<bool> has broken semantics (bit-packing,
  // no real pointers)
  static_assert(!std::is_same_v<T, bool>,
                "tensor<bool> is not supported due to std::vector<bool> "
                "specialization issues. "
                "Use tensor<uint8_t> or tensor<char> instead.");

 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  tensor() = default;

  explicit tensor(const shape_t& shape, layout_type layout = default_layout)
      : shape_(shape),
        layout_(layout),
        strides_(detail::compute_strides(shape_, layout_)),
        data_(shape_.size()) {}

  tensor(const shape_t& shape, const T& value,
         layout_type layout = default_layout)
      : shape_(shape),
        layout_(layout),
        strides_(detail::compute_strides(shape_, layout_)),
        data_(shape_.size(), value) {}

  tensor(std::initializer_list<T> data)
      : shape_({data.size()}),
        strides_(detail::compute_strides(shape_)),
        data_(data) {}

  tensor(std::initializer_list<std::initializer_list<T>> data) {
    size_type rows = data.size();
    size_type cols = rows > 0 ? data.begin()->size() : 0;

    shape_ = {rows, cols};
    strides_ = detail::compute_strides(shape_);
    data_.reserve(rows * cols);

    for (const auto& row : data) {
      TT_ASSERT(row.size() == cols, "All rows must have the same size");
      data_.insert(data_.end(), row.begin(), row.end());
    }
  }

  tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>>
             data) {
    size_type d0 = data.size();
    size_type d1 = d0 > 0 ? data.begin()->size() : 0;
    size_type d2 = d1 > 0 ? data.begin()->begin()->size() : 0;

    shape_ = {d0, d1, d2};
    strides_ = detail::compute_strides(shape_);
    data_.reserve(d0 * d1 * d2);

    for (const auto& plane : data) {
      TT_ASSERT(plane.size() == d1, "All planes must have the same size");
      for (const auto& row : plane) {
        TT_ASSERT(row.size() == d2, "All rows must have the same size");
        data_.insert(data_.end(), row.begin(), row.end());
      }
    }
  }

  template <typename View>
    requires std::is_same_v<View, tensor_view<T>> ||
                 std::is_same_v<View, tensor_view<const T>>
  explicit tensor(const View& view)
      : shape_(view.shape()), strides_(detail::compute_strides(shape_)) {
    data_.reserve(shape_.size());
    for (const auto& val : view) {
      data_.push_back(val);
    }
  }

  [[nodiscard]] size_type ndim() const noexcept { return shape_.ndim(); }
  [[nodiscard]] size_type size() const noexcept { return data_.size(); }
  [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
  [[nodiscard]] const shape_t& shape() const noexcept { return shape_; }
  [[nodiscard]] const strides_t& strides() const noexcept { return strides_; }
  [[nodiscard]] layout_type layout() const noexcept { return layout_; }
  [[nodiscard]] T* data() noexcept { return data_.data(); }
  [[nodiscard]] const T* data() const noexcept { return data_.data(); }

  [[nodiscard]] iterator begin() noexcept { return data_.begin(); }
  [[nodiscard]] iterator end() noexcept { return data_.end(); }
  [[nodiscard]] const_iterator begin() const noexcept { return data_.begin(); }
  [[nodiscard]] const_iterator end() const noexcept { return data_.end(); }

  // Variadic index access
  template <typename... Idx>
  [[nodiscard]] reference operator()(Idx... indices) {
    static_assert(sizeof...(Idx) > 0, "At least one index required");
    TT_ASSERT(sizeof...(Idx) == ndim(), "Index count must match dimensions");
    TT_ASSERT(detail::check_bounds(shape_, indices...), "Index out of bounds");
    return data_[detail::data_offset(strides_, indices...)];
  }

  template <typename... Idx>
  [[nodiscard]] const_reference operator()(Idx... indices) const {
    static_assert(sizeof...(Idx) > 0, "At least one index required");
    TT_ASSERT(sizeof...(Idx) == ndim(), "Index count must match dimensions");
    TT_ASSERT(detail::check_bounds(shape_, indices...), "Index out of bounds");
    return data_[detail::data_offset(strides_, indices...)];
  }

  // Dynamic index access via span/vector
  [[nodiscard]] reference at(std::span<const size_type> indices) {
    if (indices.size() != ndim()) {
      TT_THROW(index_error, "Index count must match dimensions");
    }
    if (!detail::check_bounds(shape_, indices)) {
      TT_THROW(index_error, "Index out of bounds");
    }
    return data_[detail::data_offset(strides_, indices)];
  }

  [[nodiscard]] const_reference at(std::span<const size_type> indices) const {
    if (indices.size() != ndim()) {
      TT_THROW(index_error, "Index count must match dimensions");
    }
    if (!detail::check_bounds(shape_, indices)) {
      TT_THROW(index_error, "Index out of bounds");
    }
    return data_[detail::data_offset(strides_, indices)];
  }

  // Flat index access
  [[nodiscard]] reference flat(size_type idx) {
    if (idx >= size()) {
      TT_THROW(index_error, "Flat index out of bounds");
    }
    return data_[idx];
  }

  [[nodiscard]] const_reference flat(size_type idx) const {
    if (idx >= size()) {
      TT_THROW(index_error, "Flat index out of bounds");
    }
    return data_[idx];
  }

  [[nodiscard]] tensor_view<T> view() {
    return tensor_view<T>(data(), 0, shape_, strides_);
  }

  [[nodiscard]] tensor_view<const T> view() const {
    return tensor_view<const T>(data(), 0, shape_, strides_);
  }

  // Reshape to new dimensions (same total size required)
  tensor& reshape(const shape_t& new_shape) {
    TT_ASSERT(new_shape.size() == size(),
              "New shape must have the same total size");
    shape_ = new_shape;
    strides_ = detail::compute_strides(shape_, layout_);
    return *this;
  }

  // Reshape with automatic dimension inference (-1 for one dimension)
  tensor& reshape(std::initializer_list<std::ptrdiff_t> dims) {
    std::vector<std::ptrdiff_t> dim_vec(dims);
    std::ptrdiff_t inferred_idx = -1;
    size_type known_product = 1;

    for (std::size_t i = 0; i < dim_vec.size(); ++i) {
      if (dim_vec[i] == -1) {
        TT_ASSERT(inferred_idx == -1, "Can only infer one dimension");
        inferred_idx = static_cast<std::ptrdiff_t>(i);
      } else {
        TT_ASSERT(dim_vec[i] > 0,
                  "Dimensions must be positive (or -1 for inference)");
        known_product *= static_cast<size_type>(dim_vec[i]);
      }
    }

    shape_t new_shape;
    new_shape.reserve(dim_vec.size());

    if (inferred_idx >= 0) {
      TT_ASSERT(size() % known_product == 0,
                "Cannot infer dimension: size not divisible");
      size_type inferred_size = size() / known_product;
      for (std::size_t i = 0; i < dim_vec.size(); ++i) {
        if (static_cast<std::ptrdiff_t>(i) == inferred_idx) {
          new_shape.push_back(inferred_size);
        } else {
          new_shape.push_back(static_cast<size_type>(dim_vec[i]));
        }
      }
    } else {
      for (auto d : dim_vec) {
        new_shape.push_back(static_cast<size_type>(d));
      }
    }

    return reshape(new_shape);
  }

  // Resize (changes total size, may lose or gain data)
  tensor& resize(const shape_t& new_shape) {
    shape_ = new_shape;
    strides_ = detail::compute_strides(shape_, layout_);
    data_.resize(shape_.size());
    return *this;
  }

  tensor& resize(const shape_t& new_shape, const T& fill_value) {
    size_type old_size = data_.size();
    shape_ = new_shape;
    strides_ = detail::compute_strides(shape_, layout_);
    data_.resize(shape_.size(), fill_value);
    return *this;
  }

  // Flatten to 1D
  tensor& flatten() {
    shape_ = shape_t{size()};
    strides_ = detail::compute_strides(shape_, layout_);
    return *this;
  }

  // Return flattened copy
  [[nodiscard]] tensor flattened() const {
    tensor result = *this;
    result.flatten();
    return result;
  }

  // Squeeze: remove dimensions of size 1
  tensor& squeeze() {
    shape_t new_shape;
    for (size_type i = 0; i < shape_.ndim(); ++i) {
      if (shape_[i] != 1) {
        new_shape.push_back(shape_[i]);
      }
    }
    if (new_shape.empty() && size() > 0) {
      new_shape.push_back(1);  // Keep at least one dimension
    }
    shape_ = std::move(new_shape);
    strides_ = detail::compute_strides(shape_, layout_);
    return *this;
  }

  // Unsqueeze: add dimension of size 1 at position
  tensor& unsqueeze(size_type axis) {
    TT_ASSERT(axis <= ndim(), "Axis out of bounds");
    shape_t new_shape;
    new_shape.reserve(ndim() + 1);
    for (size_type i = 0; i < ndim(); ++i) {
      if (i == axis) new_shape.push_back(1);
      new_shape.push_back(shape_[i]);
    }
    if (axis == ndim()) new_shape.push_back(1);
    shape_ = std::move(new_shape);
    strides_ = detail::compute_strides(shape_, layout_);
    return *this;
  }

  // Transpose (reverse dimensions)
  [[nodiscard]] tensor transposed() const {
    if (ndim() < 2) return *this;

    shape_t new_shape;
    new_shape.reserve(ndim());
    for (size_type i = ndim(); i > 0; --i) {
      new_shape.push_back(shape_[i - 1]);
    }

    tensor result(new_shape);

    std::vector<size_type> src_idx(ndim(), 0);
    std::vector<size_type> dst_idx(ndim(), 0);

    for (size_type flat = 0; flat < size(); ++flat) {
      // Convert flat to source indices
      size_type tmp = flat;
      for (size_type d = ndim(); d > 0; --d) {
        src_idx[d - 1] = tmp % shape_[d - 1];
        tmp /= shape_[d - 1];
      }
      // Reverse for destination
      for (size_type d = 0; d < ndim(); ++d) {
        dst_idx[d] = src_idx[ndim() - 1 - d];
      }
      result.at(dst_idx) = at(src_idx);
    }

    return result;
  }

  void fill(const T& value) { std::fill(data_.begin(), data_.end(), value); }

  [[nodiscard]] bool is_contiguous() const noexcept {
    return true;  // Owning tensor is always contiguous
  }

  [[nodiscard]] broadcast_view<T> broadcast_to(const shape_t& target_shape) {
    return tt::broadcast(data(), 0, shape_, strides_, target_shape);
  }

  [[nodiscard]] broadcast_view<const T> broadcast_to(
      const shape_t& target_shape) const {
    return tt::broadcast(data(), 0, shape_, strides_, target_shape);
  }

  // Copy data from another tensor (shapes must match)
  template <typename U>
  tensor& copy_from(const tensor<U>& other) {
    TT_ASSERT(shape_ == other.shape(), "Shapes must match for copy");
    if (layout_ == other.layout()) {
      std::copy(other.begin(), other.end(), data_.begin());
    } else {
      auto src = other.view();
      auto dst = this->view();
      auto si = src.begin();
      auto di = dst.begin();
      for (; di != dst.end(); ++si, ++di) {
        *di = *si;
      }
    }
    return *this;
  }

  // Deep copy
  [[nodiscard]] tensor copy() const { return *this; }

 private:
  shape_t shape_;
  layout_type layout_ = default_layout;
  strides_t strides_;
  std::vector<T> data_;
};

template <typename T, typename... Slices>
auto view(tensor<T>& t, Slices&&... slices) {
  std::vector<slice_t> slice_vec{slice_t{std::forward<Slices>(slices)}...};
  return detail::apply_slices(t.data(), 0, t.shape(), t.strides(), slice_vec);
}

template <typename T, typename... Slices>
auto view(const tensor<T>& t, Slices&&... slices) {
  std::vector<slice_t> slice_vec{slice_t{std::forward<Slices>(slices)}...};
  return detail::apply_slices(const_cast<const T*>(t.data()), 0, t.shape(),
                              t.strides(), slice_vec);
}

template <typename T, typename... Slices>
auto view(tensor_view<T>& v, Slices&&... slices) {
  std::vector<slice_t> slice_vec{slice_t{std::forward<Slices>(slices)}...};
  return detail::apply_slices(v.data(), v.offset(), v.shape(), v.strides(),
                              slice_vec);
}

template <typename T, typename... Slices>
auto view(const tensor_view<T>& v, Slices&&... slices) {
  std::vector<slice_t> slice_vec{slice_t{std::forward<Slices>(slices)}...};
  return detail::apply_slices(const_cast<const T*>(v.data()), v.offset(),
                              v.shape(), v.strides(), slice_vec);
}

template <typename T>
tensor<T> zeros(const shape_t& shape) {
  return tensor<T>(shape, T{0});
}

template <typename T>
tensor<T> ones(const shape_t& shape) {
  return tensor<T>(shape, T{1});
}

template <typename T>
tensor<T> full(const shape_t& shape, const T& value) {
  return tensor<T>(shape, value);
}

template <typename T>
tensor<T> arange(T start, T stop, T step = T{1}) {
  if (step == T{0}) {
    TT_THROW(index_error, "Step cannot be zero");
  }

  std::size_t count = 0;
  if (step > T{0} && start < stop) {
    if constexpr (std::is_floating_point_v<T>) {
      count = static_cast<std::size_t>(std::ceil((stop - start) / step));
    } else {
      count = static_cast<std::size_t>((stop - start + step - T{1}) / step);
    }
  } else if (step < T{0} && start > stop) {
    if constexpr (std::is_floating_point_v<T>) {
      count = static_cast<std::size_t>(std::ceil((start - stop) / (-step)));
    } else {
      count =
          static_cast<std::size_t>((start - stop - step - T{1}) / (-step));
    }
  }

  tensor<T> result(shape_t{count});
  for (std::size_t i = 0; i < count; ++i) {
    result.flat(i) = start + static_cast<T>(i) * step;
  }
  return result;
}

template <typename T>
tensor<T> linspace(T start, T stop, std::size_t num) {
  TT_ASSERT(num > 0, "Number of samples must be positive");

  tensor<T> result(shape_t{num});
  if (num == 1) {
    result(0) = start;
    return result;
  }

  T step = (stop - start) / static_cast<T>(num - 1);
  for (std::size_t i = 0; i < num; ++i) {
    result(i) = start + static_cast<T>(i) * step;
  }
  return result;
}

// Create identity matrix
template <typename T>
tensor<T> eye(std::size_t n) {
  tensor<T> result(shape_t{n, n}, T{0});
  for (std::size_t i = 0; i < n; ++i) {
    result(i, i) = T{1};
  }
  return result;
}

// Create diagonal matrix from 1D tensor
template <typename T>
tensor<T> diag(const tensor<T>& v) {
  TT_ASSERT(v.ndim() == 1, "Input must be 1D");
  std::size_t n = v.size();
  tensor<T> result(shape_t{n, n}, T{0});
  for (std::size_t i = 0; i < n; ++i) {
    result(i, i) = v(i);
  }
  return result;
}

}  // namespace tt
