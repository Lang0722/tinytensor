#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

#include "../core/concepts.hpp"
#include "../views/broadcast.hpp"
#include "tensor.hpp"

namespace tt {

namespace detail {

template <typename T, typename U, typename Op>
auto binary_op(const tensor<T>& a, const tensor<U>& b, Op op) {
  using result_type = decltype(op(std::declval<T>(), std::declval<U>()));

  shape_t result_shape;
  if (!broadcast_shapes(a.shape(), b.shape(), result_shape)) {
    TT_THROW(broadcast_error, "Cannot broadcast shapes for binary operation");
  }

  layout_type result_layout =
      (a.layout() == b.layout()) ? a.layout() : default_layout;
  tensor<result_type> result(result_shape, result_layout);

  auto a_bc = a.broadcast_to(result_shape);
  auto b_bc = b.broadcast_to(result_shape);

  auto a_it = a_bc.begin();
  auto b_it = b_bc.begin();
  auto r_view = result.view();
  auto r_it = r_view.begin();

  for (; r_it != r_view.end(); ++a_it, ++b_it, ++r_it) {
    *r_it = op(*a_it, *b_it);
  }

  return result;
}

template <typename T, typename Scalar, typename Op>
auto binary_op_scalar(const tensor<T>& a, const Scalar& s, Op op) {
  using result_type = decltype(op(std::declval<T>(), std::declval<Scalar>()));

  tensor<result_type> result(a.shape(), a.layout());

  auto a_it = a.begin();
  auto r_it = result.begin();

  for (; r_it != result.end(); ++a_it, ++r_it) {
    *r_it = op(*a_it, s);
  }

  return result;
}

template <typename T, typename Scalar, typename Op>
auto scalar_binary_op(const Scalar& s, const tensor<T>& a, Op op) {
  using result_type = decltype(op(std::declval<Scalar>(), std::declval<T>()));

  tensor<result_type> result(a.shape(), a.layout());

  auto a_it = a.begin();
  auto r_it = result.begin();

  for (; r_it != result.end(); ++a_it, ++r_it) {
    *r_it = op(s, *a_it);
  }

  return result;
}

template <typename T, typename Op>
auto unary_op(const tensor<T>& a, Op op) {
  using result_type = decltype(op(std::declval<T>()));

  tensor<result_type> result(a.shape(), a.layout());

  std::transform(a.begin(), a.end(), result.begin(), op);

  return result;
}

}  // namespace detail

// Arithmetic operators: tensor + tensor
template <typename T, typename U>
auto operator+(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(a, b, std::plus<>{});
}

template <typename T, typename U>
auto operator-(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(a, b, std::minus<>{});
}

template <typename T, typename U>
auto operator*(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(a, b, std::multiplies<>{});
}

template <typename T, typename U>
auto operator/(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(a, b, std::divides<>{});
}

// Arithmetic operators: tensor + scalar
template <typename T, arithmetic S>
auto operator+(const tensor<T>& a, const S& s) {
  return detail::binary_op_scalar(a, s, std::plus<>{});
}

template <typename T, arithmetic S>
auto operator-(const tensor<T>& a, const S& s) {
  return detail::binary_op_scalar(a, s, std::minus<>{});
}

template <typename T, arithmetic S>
auto operator*(const tensor<T>& a, const S& s) {
  return detail::binary_op_scalar(a, s, std::multiplies<>{});
}

template <typename T, arithmetic S>
auto operator/(const tensor<T>& a, const S& s) {
  return detail::binary_op_scalar(a, s, std::divides<>{});
}

// Arithmetic operators: scalar + tensor
template <arithmetic S, typename T>
auto operator+(const S& s, const tensor<T>& a) {
  return detail::scalar_binary_op(s, a, std::plus<>{});
}

template <arithmetic S, typename T>
auto operator-(const S& s, const tensor<T>& a) {
  return detail::scalar_binary_op(s, a, std::minus<>{});
}

template <arithmetic S, typename T>
auto operator*(const S& s, const tensor<T>& a) {
  return detail::scalar_binary_op(s, a, std::multiplies<>{});
}

template <arithmetic S, typename T>
auto operator/(const S& s, const tensor<T>& a) {
  return detail::scalar_binary_op(s, a, std::divides<>{});
}

// Unary operators
template <typename T>
auto operator-(const tensor<T>& a) {
  return detail::unary_op(a, std::negate<>{});
}

// Math functions
template <typename T>
auto abs(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::abs(x); });
}

template <typename T>
auto sqrt(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::sqrt(x); });
}

template <typename T>
auto exp(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::exp(x); });
}

template <typename T>
auto log(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::log(x); });
}

template <typename T>
auto sin(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::sin(x); });
}

template <typename T>
auto cos(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::cos(x); });
}

template <typename T>
auto tan(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::tan(x); });
}

template <typename T>
auto asin(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::asin(x); });
}

template <typename T>
auto acos(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::acos(x); });
}

template <typename T>
auto atan(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::atan(x); });
}

template <typename T>
auto sinh(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::sinh(x); });
}

template <typename T>
auto cosh(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::cosh(x); });
}

template <typename T>
auto tanh(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::tanh(x); });
}

template <typename T>
auto floor(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::floor(x); });
}

template <typename T>
auto ceil(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::ceil(x); });
}

template <typename T>
auto round(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::round(x); });
}

template <typename T>
auto log10(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::log10(x); });
}

template <typename T>
auto log2(const tensor<T>& a) {
  return detail::unary_op(a, [](const T& x) { return std::log2(x); });
}

template <typename T, typename U>
auto pow(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(
      a, b, [](const T& x, const U& y) { return std::pow(x, y); });
}

template <typename T, arithmetic S>
auto pow(const tensor<T>& a, const S& s) {
  return detail::binary_op_scalar(
      a, s, [](const T& x, const S& y) { return std::pow(x, y); });
}

// Reduction operations
template <typename T>
T sum(const tensor<T>& a) {
  T result = T{0};
  for (const auto& val : a) {
    result += val;
  }
  return result;
}

template <typename T>
T prod(const tensor<T>& a) {
  T result = T{1};
  for (const auto& val : a) {
    result *= val;
  }
  return result;
}

template <typename T>
T min(const tensor<T>& a) {
  TT_ASSERT(!a.empty(), "Cannot compute min of empty tensor");
  return *std::min_element(a.begin(), a.end());
}

template <typename T>
T max(const tensor<T>& a) {
  TT_ASSERT(!a.empty(), "Cannot compute max of empty tensor");
  return *std::max_element(a.begin(), a.end());
}

template <typename T>
auto mean(const tensor<T>& a) {
  TT_ASSERT(!a.empty(), "Cannot compute mean of empty tensor");
  using result_type = std::conditional_t<std::is_integral_v<T>, double, T>;
  return static_cast<result_type>(sum(a)) / static_cast<result_type>(a.size());
}

// Comparison operators (element-wise, return tensor<uint8_t> since tensor<bool>
// is not supported)
template <typename T, typename U>
tensor<uint8_t> operator==(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(
      a, b, [](const T& x, const U& y) -> uint8_t { return x == y ? 1 : 0; });
}

template <typename T, typename U>
tensor<uint8_t> operator!=(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(
      a, b, [](const T& x, const U& y) -> uint8_t { return x != y ? 1 : 0; });
}

template <typename T, typename U>
tensor<uint8_t> operator<(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(
      a, b, [](const T& x, const U& y) -> uint8_t { return x < y ? 1 : 0; });
}

template <typename T, typename U>
tensor<uint8_t> operator<=(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(
      a, b, [](const T& x, const U& y) -> uint8_t { return x <= y ? 1 : 0; });
}

template <typename T, typename U>
tensor<uint8_t> operator>(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(
      a, b, [](const T& x, const U& y) -> uint8_t { return x > y ? 1 : 0; });
}

template <typename T, typename U>
tensor<uint8_t> operator>=(const tensor<T>& a, const tensor<U>& b) {
  return detail::binary_op(
      a, b, [](const T& x, const U& y) -> uint8_t { return x >= y ? 1 : 0; });
}

// All/any reductions for boolean tensors (using uint8_t where non-zero is true)
inline bool all_of(const tensor<uint8_t>& a) {
  return std::all_of(a.begin(), a.end(), [](uint8_t v) { return v != 0; });
}

inline bool any_of(const tensor<uint8_t>& a) {
  return std::any_of(a.begin(), a.end(), [](uint8_t v) { return v != 0; });
}

}  // namespace tt
