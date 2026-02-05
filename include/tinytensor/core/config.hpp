#pragma once

#define TINYTENSOR_VERSION_MAJOR 0
#define TINYTENSOR_VERSION_MINOR 1
#define TINYTENSOR_VERSION_PATCH 0

#define TINYTENSOR_VERSION                                             \
  (TINYTENSOR_VERSION_MAJOR * 10000 + TINYTENSOR_VERSION_MINOR * 100 + \
   TINYTENSOR_VERSION_PATCH)

namespace tt {

inline constexpr int version_major = TINYTENSOR_VERSION_MAJOR;
inline constexpr int version_minor = TINYTENSOR_VERSION_MINOR;
inline constexpr int version_patch = TINYTENSOR_VERSION_PATCH;

}  // namespace tt
