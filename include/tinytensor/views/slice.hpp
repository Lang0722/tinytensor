#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <variant>

#include "../utils/assert.hpp"

namespace tt {

inline constexpr std::ptrdiff_t end_marker =
    std::numeric_limits<std::ptrdiff_t>::max();

struct range_t {
  std::ptrdiff_t start = 0;
  std::ptrdiff_t stop = end_marker;
  std::ptrdiff_t step = 1;

  constexpr range_t() = default;
  constexpr range_t(std::ptrdiff_t start_, std::ptrdiff_t stop_,
                    std::ptrdiff_t step_ = 1)
      : start(start_), stop(stop_), step(step_) {}
};

struct all_tag {};
struct newaxis_tag {};
struct ellipsis_tag {};

inline constexpr all_tag all{};
inline constexpr newaxis_tag newaxis{};
inline constexpr ellipsis_tag ellipsis{};
inline constexpr std::ptrdiff_t _ = end_marker;

using slice_t =
    std::variant<std::ptrdiff_t, range_t, all_tag, newaxis_tag, ellipsis_tag>;

constexpr range_t range(std::ptrdiff_t start, std::ptrdiff_t stop,
                        std::ptrdiff_t step = 1) {
  return {start, stop, step};
}

constexpr range_t range(std::ptrdiff_t stop) { return {0, stop, 1}; }

namespace detail {

constexpr std::ptrdiff_t normalize_index(std::ptrdiff_t idx, std::size_t size) {
  if (idx < 0) {
    idx += static_cast<std::ptrdiff_t>(size);
  }
  return idx;
}

constexpr std::ptrdiff_t clamp_index(std::ptrdiff_t idx, std::size_t size) {
  if (idx < 0) return 0;
  if (idx > static_cast<std::ptrdiff_t>(size))
    return static_cast<std::ptrdiff_t>(size);
  return idx;
}

struct normalized_range {
  std::size_t start;
  std::size_t size;
  std::ptrdiff_t step;
};

inline normalized_range normalize_range(const range_t& r,
                                        std::size_t dim_size) {
  if (r.step == 0) {
    TT_THROW(index_error, "Slice step cannot be zero");
  }

  std::ptrdiff_t ssize = static_cast<std::ptrdiff_t>(dim_size);
  std::ptrdiff_t step = r.step;

  std::ptrdiff_t start = r.start;
  std::ptrdiff_t stop = r.stop;

  if (step > 0) {
    // Positive step: normalize negative indices
    if (start < 0) start += ssize;
    if (stop == end_marker) {
      stop = ssize;
    } else if (stop < 0) {
      stop += ssize;
    }
    start = clamp_index(start, dim_size);
    stop = clamp_index(stop, dim_size);
    std::size_t size =
        (stop > start)
            ? static_cast<std::size_t>((stop - start + step - 1) / step)
            : 0;
    return {static_cast<std::size_t>(start), size, step};
  } else {
    // Negative step: iterate from start down to (but not including) stop
    // For negative step, stop=-1 means "include index 0", so we don't normalize
    // it
    if (start == end_marker || start >= ssize) {
      start = ssize - 1;
    } else if (start < 0) {
      start += ssize;
    }
    if (start < 0) return {0, 0, step};

    // stop == end_marker means go all the way to the beginning
    if (stop == end_marker) {
      stop = -1;
    } else if (stop < 0) {
      stop += ssize;
      if (stop < -1) stop = -1;
    }

    std::size_t size =
        (start > stop)
            ? static_cast<std::size_t>((start - stop + (-step) - 1) / (-step))
            : 0;
    return {static_cast<std::size_t>(start), size, step};
  }
}

}  // namespace detail

}  // namespace tt
