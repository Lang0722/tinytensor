#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "../core/layout.hpp"
#include "../utils/assert.hpp"

namespace tt {

namespace detail {

template <std::size_t... Dims>
constexpr std::size_t compute_size() {
  return (Dims * ... * 1);
}

template <std::size_t... Dims>
constexpr std::array<std::size_t, sizeof...(Dims)> make_shape_array() {
  return {Dims...};
}

template <std::size_t... Dims>
constexpr std::array<std::ptrdiff_t, sizeof...(Dims)> compute_fixed_strides() {
  constexpr std::size_t ndim = sizeof...(Dims);
  std::array<std::size_t, ndim> shape = {Dims...};
  std::array<std::ptrdiff_t, ndim> strides{};

  std::ptrdiff_t data_size = 1;
  for (std::size_t i = ndim; i != 0; --i) {
    strides[i - 1] = (shape[i - 1] == 1) ? 0 : data_size;
    data_size *= static_cast<std::ptrdiff_t>(shape[i - 1]);
  }

  return strides;
}

template <typename Strides, typename... Idx>
constexpr std::size_t fixed_data_offset(const Strides& strides,
                                        Idx... indices) noexcept {
  std::ptrdiff_t offset = 0;
  std::size_t i = 0;
  ((offset += strides[i++] * static_cast<std::ptrdiff_t>(indices)), ...);
  return static_cast<std::size_t>(offset);
}

}  // namespace detail

template <typename T, std::size_t... Dims>
class fixed_tensor {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  static constexpr std::size_t ndim_v = sizeof...(Dims);
  static constexpr std::size_t size_v = detail::compute_size<Dims...>();
  static constexpr std::array<std::size_t, ndim_v> shape_v =
      detail::make_shape_array<Dims...>();
  static constexpr std::array<std::ptrdiff_t, ndim_v> strides_v =
      detail::compute_fixed_strides<Dims...>();

  using iterator = typename std::array<T, size_v>::iterator;
  using const_iterator = typename std::array<T, size_v>::const_iterator;

  constexpr fixed_tensor() = default;

  constexpr explicit fixed_tensor(const T& value) { data_.fill(value); }

  constexpr fixed_tensor(std::initializer_list<T> init) {
    TT_ASSERT(init.size() == size_v,
              "Initializer list size must match tensor size");
    std::copy(init.begin(), init.end(), data_.begin());
  }

  [[nodiscard]] static constexpr size_type ndim() noexcept { return ndim_v; }
  [[nodiscard]] static constexpr size_type size() noexcept { return size_v; }
  [[nodiscard]] static constexpr const auto& shape() noexcept {
    return shape_v;
  }
  [[nodiscard]] static constexpr const auto& strides() noexcept {
    return strides_v;
  }
  [[nodiscard]] constexpr T* data() noexcept { return data_.data(); }
  [[nodiscard]] constexpr const T* data() const noexcept {
    return data_.data();
  }

  [[nodiscard]] constexpr iterator begin() noexcept { return data_.begin(); }
  [[nodiscard]] constexpr iterator end() noexcept { return data_.end(); }
  [[nodiscard]] constexpr const_iterator begin() const noexcept {
    return data_.begin();
  }
  [[nodiscard]] constexpr const_iterator end() const noexcept {
    return data_.end();
  }
  [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
    return data_.cbegin();
  }
  [[nodiscard]] constexpr const_iterator cend() const noexcept {
    return data_.cend();
  }

  template <typename... Idx>
  [[nodiscard]] constexpr reference operator()(Idx... indices) {
    static_assert(sizeof...(Idx) == ndim_v,
                  "Index count must match dimensions");
    return data_[detail::fixed_data_offset(strides_v, indices...)];
  }

  template <typename... Idx>
  [[nodiscard]] constexpr const_reference operator()(Idx... indices) const {
    static_assert(sizeof...(Idx) == ndim_v,
                  "Index count must match dimensions");
    return data_[detail::fixed_data_offset(strides_v, indices...)];
  }

  constexpr void fill(const T& value) { data_.fill(value); }

  [[nodiscard]] static constexpr bool is_contiguous() noexcept { return true; }

 private:
  std::array<T, size_v> data_{};
};

template <typename T, std::size_t... Dims>
constexpr fixed_tensor<T, Dims...> zeros_fixed() {
  return fixed_tensor<T, Dims...>(T{0});
}

template <typename T, std::size_t... Dims>
constexpr fixed_tensor<T, Dims...> ones_fixed() {
  return fixed_tensor<T, Dims...>(T{1});
}

template <typename T, std::size_t... Dims>
constexpr fixed_tensor<T, Dims...> full_fixed(const T& value) {
  return fixed_tensor<T, Dims...>(value);
}

}  // namespace tt
