#pragma once

#include <algorithm>
#include <compare>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <span>
#include <vector>

namespace tt {

class shape_t {
 public:
  using value_type = std::size_t;
  using container_type = std::vector<value_type>;
  using iterator = container_type::iterator;
  using const_iterator = container_type::const_iterator;

  shape_t() = default;

  shape_t(std::initializer_list<value_type> dims) : dims_(dims) {}

  explicit shape_t(std::span<const value_type> dims)
      : dims_(dims.begin(), dims.end()) {}

  template <typename Iter>
  shape_t(Iter first, Iter last) : dims_(first, last) {}

  [[nodiscard]] std::size_t ndim() const noexcept { return dims_.size(); }

  [[nodiscard]] std::size_t size() const noexcept {
    if (dims_.empty()) return 0;
    return std::accumulate(dims_.begin(), dims_.end(), std::size_t{1},
                           std::multiplies<>{});
  }

  [[nodiscard]] bool empty() const noexcept { return dims_.empty(); }

  [[nodiscard]] value_type operator[](std::size_t i) const { return dims_[i]; }
  [[nodiscard]] value_type& operator[](std::size_t i) { return dims_[i]; }

  [[nodiscard]] const value_type* data() const noexcept { return dims_.data(); }
  [[nodiscard]] value_type* data() noexcept { return dims_.data(); }

  [[nodiscard]] iterator begin() noexcept { return dims_.begin(); }
  [[nodiscard]] iterator end() noexcept { return dims_.end(); }
  [[nodiscard]] const_iterator begin() const noexcept { return dims_.begin(); }
  [[nodiscard]] const_iterator end() const noexcept { return dims_.end(); }
  [[nodiscard]] const_iterator cbegin() const noexcept {
    return dims_.cbegin();
  }
  [[nodiscard]] const_iterator cend() const noexcept { return dims_.cend(); }

  void push_back(value_type dim) { dims_.push_back(dim); }
  void resize(std::size_t n) { dims_.resize(n); }
  void reserve(std::size_t n) { dims_.reserve(n); }

  [[nodiscard]] auto operator<=>(const shape_t&) const = default;

 private:
  container_type dims_;
};

}  // namespace tt
