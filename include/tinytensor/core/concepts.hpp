#pragma once

#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

namespace tt {

template <typename T>
concept arithmetic = std::integral<T> || std::floating_point<T>;

template <typename T>
concept tensor_like = requires(T t, std::size_t i) {
  { t.shape() } -> std::ranges::range;
  { t.size() } -> std::same_as<std::size_t>;
  { t.data() };
  { t.ndim() } -> std::same_as<std::size_t>;
};

template <typename T>
concept view_like = tensor_like<T> && requires(T t) {
  { t.offset() } -> std::same_as<std::ptrdiff_t>;
  { t.strides() };
};

template <typename T>
concept contiguous_tensor = tensor_like<T> && requires(T t) {
  { t.is_contiguous() } -> std::same_as<bool>;
};

}  // namespace tt
