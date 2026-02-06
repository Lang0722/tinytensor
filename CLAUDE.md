# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run all tests via CTest
ctest --test-dir build

# Run individual test binaries
./build/tests/tensor_test
./build/tests/xtensor_compat_test

# Run a single test case (gtest filter)
./build/tests/tensor_test --gtest_filter="TensorSlice.BasicRange"

# Format code
clang-format -i include/tinytensor/**/*.hpp  # .clang-format uses Google style
```

Requires: C++20 compiler, CMake 3.16+, Google Test (gtest).

## Architecture

Header-only C++20 tensor library. Everything lives in `include/tinytensor/`, namespace `tt`. Single umbrella header: `tinytensor/tinytensor.hpp`.

### Module Layout

- **`core/`** - Foundational types: `shape_t` (dynamic shape wrapping `vector<size_t>`), `strides_t` (`vector<ptrdiff_t>`, signed for negative-stride views), `layout_type` enum, C++20 concepts (`tensor_like`, `view_like`, `contiguous_tensor`)
- **`tensor/`** - `tensor<T>` (owning, `std::vector<T>` storage), `tensor_ops.hpp` (element-wise ops, reductions, broadcasting operators)
- **`views/`** - `tensor_view<T>` (non-owning strided view), `slice.hpp` (Python-style slicing), `broadcast_view<T>` (read-only broadcasting)
- **`utils/`** - Exception hierarchy (`tensor_error` > `shape_error`/`index_error`/`broadcast_error`), `TT_ASSERT` (debug-only) and `TT_THROW` (always-on) macros

### Key Design Decisions

**Iterator semantics differ between tensor and tensor_view:**
- `tensor::begin()/end()` are raw `std::vector::iterator` — they iterate in **memory order** (flat)
- `tensor_view::iterator` uses multi-index tracking — iterates in **logical row-major order** regardless of strides

**Slicing sentinel:** `tt::_` (aliased to `end_marker`, which is `ptrdiff_t::max()`) means "go to end" in ranges. Do NOT use `-1` as end sentinel — `-1` normalizes to `size - 1` following Python semantics.

**`tensor<bool>` is disabled** via `static_assert` because `std::vector<bool>` bit-packing breaks pointer semantics. Use `tensor<uint8_t>` instead. Comparison operators return `tensor<uint8_t>`.

**Layout:** Default is `row_major`. Column-major supported via constructor parameter. Operations (`unary_op`, `binary_op_scalar`, `scalar_binary_op`) preserve the layout of the input tensor.

**Evaluation model:** Eager evaluation (no expression templates or lazy computation). Broadcasting is done via stride-0 trick (no memory duplication).

**Bounds checking:** `TT_ASSERT` compiles to `(void)0` under `NDEBUG`. `TT_THROW` is always active for invariant violations.

### Tensor vs View Relationship

`tensor<T>` owns contiguous memory. `tensor_view<T>` holds a raw `T*` pointer + offset + shape + strides into a tensor's buffer. Views are created via `tt::view(tensor, slices...)` or `tensor.view()`. Views can be chained (view of a view still points to original tensor data). `tensor_view` can be materialized back to an owning `tensor` via the `tensor(tensor_view&)` constructor. `copy_from` handles layout mismatches by using view iterators.

### Broadcasting

Binary operations auto-broadcast using NumPy rules (right-align shapes, dimensions must be equal or one must be 1). The `detail::broadcast_shapes()` function computes the result shape, and `detail::broadcast_strides()` sets stride=0 for broadcast dimensions.

### Slice Processing

`detail::apply_slices()` in `tensor_view.hpp` processes `slice_t` variants: `range_t` selects subranges, integer indices collapse dimensions (reducing ndim), `all_tag` keeps dimensions, `newaxis_tag` inserts size-1 dimensions, `ellipsis_tag` expands to fill remaining dimensions. Negative indices in ranges normalize via `detail::normalize_index()`.
