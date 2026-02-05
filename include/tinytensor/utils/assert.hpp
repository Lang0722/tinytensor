#pragma once

#include <stdexcept>
#include <string>

namespace tt {

class tensor_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class shape_error : public tensor_error {
 public:
  using tensor_error::tensor_error;
};

class index_error : public tensor_error {
 public:
  using tensor_error::tensor_error;
};

class broadcast_error : public tensor_error {
 public:
  using tensor_error::tensor_error;
};

namespace detail {

inline void assert_impl(bool condition, const char* message, const char* file,
                        int line) {
  if (!condition) [[unlikely]] {
    throw tensor_error(std::string(message) + " at " + file + ":" +
                       std::to_string(line));
  }
}

}  // namespace detail

#ifdef NDEBUG
#define TT_ASSERT(cond, msg) ((void)0)
#else
#define TT_ASSERT(cond, msg) \
  ::tt::detail::assert_impl((cond), (msg), __FILE__, __LINE__)
#endif

#define TT_THROW(ExceptionType, msg) throw ExceptionType(msg)

}  // namespace tt
