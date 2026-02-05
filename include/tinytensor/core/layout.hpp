#pragma once

namespace tt {

enum class layout_type { row_major, column_major };

inline constexpr layout_type default_layout = layout_type::row_major;

}  // namespace tt
