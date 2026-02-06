# tinytensor Code Documentation

**Version:** 0.1.0
**Language:** C++20
**Namespace:** `tt`
**License:** Header-only library

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Build System](#build-system)
4. [Core Module](#core-module)
   - [config.hpp](#confighpp)
   - [layout.hpp](#layouthpp)
   - [shape.hpp](#shapehpp)
   - [strides.hpp](#strideshpp)
   - [concepts.hpp](#conceptshpp)
5. [Tensor Module](#tensor-module)
   - [tensor.hpp](#tensorhpp)
   - [tensor_ops.hpp](#tensor_opshpp)
6. [Views Module](#views-module)
   - [slice.hpp](#slicehpp)
   - [tensor_view.hpp](#tensor_viewhpp)
   - [broadcast.hpp](#broadcasthpp)
7. [Utilities Module](#utilities-module)
   - [assert.hpp](#asserthpp)
8. [API Reference](#api-reference)
9. [Design Decisions](#design-decisions)

---

## Overview

tinytensor is a lightweight, header-only C++20 tensor library that provides NumPy-style multidimensional array operations. It supports dynamic and compile-time-fixed tensors, strided views, broadcasting, slicing, and element-wise mathematical operations.

Key features:

- **Header-only**: Include `tinytensor/tinytensor.hpp` and use it immediately.
- **C++20 concepts**: Type constraints for tensor-like objects.
- **Dynamic tensors**: `tt::tensor<T>` with runtime-determined shape.
- **NumPy-style slicing**: `range`, `all`, `newaxis`, `ellipsis`, negative indices.
- **Broadcasting**: Automatic shape broadcasting for binary operations.
- **Element-wise math**: Arithmetic operators, trigonometric, exponential, logarithmic, and reduction functions.

### Single-header entry point

```cpp
#include <tinytensor/tinytensor.hpp>
```

This umbrella header includes all modules.

---

## Project Structure

```
tinytensor/
├── CMakeLists.txt                          # Root build configuration
├── cmake/
│   └── tinytensor-config.cmake.in          # CMake package config template
├── include/
│   └── tinytensor/
│       ├── tinytensor.hpp                  # Umbrella header
│       ├── core/
│       │   ├── config.hpp                  # Version macros and constants
│       │   ├── concepts.hpp                # C++20 concept definitions
│       │   ├── layout.hpp                  # Memory layout enum
│       │   ├── shape.hpp                   # shape_t class
│       │   └── strides.hpp                 # Stride computation utilities
│       ├── tensor/
│       │   ├── tensor.hpp                  # Dynamic tensor + factory functions
│       │   └── tensor_ops.hpp              # Operators and math functions
│       ├── utils/
│       │   └── assert.hpp                  # Assertion macros and exception types
│       └── views/
│           ├── slice.hpp                   # Slice types (range, all, newaxis, ellipsis)
│           ├── tensor_view.hpp             # Non-owning strided view
│           └── broadcast.hpp               # Broadcasting view and utilities
└── tests/
    ├── CMakeLists.txt                      # Test build configuration
    ├── tensor_test.cpp                     # Core library tests
    └── xtensor_compat_test.cpp             # xtensor-compatible API tests
```

### Dependency graph

```
tinytensor.hpp
├── core/config.hpp
├── core/concepts.hpp
├── core/layout.hpp
├── core/shape.hpp
├── core/strides.hpp          ← depends on layout.hpp, shape.hpp
├── utils/assert.hpp
├── views/slice.hpp           ← depends on assert.hpp
├── views/tensor_view.hpp     ← depends on shape.hpp, strides.hpp, assert.hpp, slice.hpp
├── views/broadcast.hpp       ← depends on shape.hpp, strides.hpp, assert.hpp
├── tensor/tensor.hpp         ← depends on core/*, views/*, utils/*
└── tensor/tensor_ops.hpp     ← depends on concepts.hpp, broadcast.hpp, tensor.hpp
```

---

## Build System

### CMake configuration

The project uses CMake (minimum 3.16) and requires a C++20-compliant compiler.

```cmake
target_compile_features(tinytensor INTERFACE cxx_std_20)
```

### Build options

| Option                   | Default                | Description                    |
|--------------------------|------------------------|--------------------------------|
| `TINYTENSOR_BUILD_TESTS` | `ON` (standalone only) | Build the Google Test suite     |
| `TINYTENSOR_INSTALL`     | `ON` (standalone only) | Generate install/export targets |

### Standalone build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

### As a subproject

```cmake
add_subdirectory(tinytensor)
target_link_libraries(my_target PRIVATE tinytensor::tinytensor)
```

### System install

```bash
cmake --install build --prefix /usr/local
```

After installation, downstream projects use:

```cmake
find_package(tinytensor REQUIRED)
target_link_libraries(my_target PRIVATE tinytensor::tinytensor)
```

---

## Core Module

### config.hpp

**File:** `include/tinytensor/core/config.hpp`

Defines version information as both preprocessor macros and `constexpr` variables.

| Symbol                        | Type             | Value   |
|-------------------------------|------------------|---------|
| `TINYTENSOR_VERSION_MAJOR`    | macro            | `0`     |
| `TINYTENSOR_VERSION_MINOR`    | macro            | `1`     |
| `TINYTENSOR_VERSION_PATCH`    | macro            | `0`     |
| `TINYTENSOR_VERSION`          | macro            | `100`   |
| `tt::version_major`           | `constexpr int`  | `0`     |
| `tt::version_minor`           | `constexpr int`  | `1`     |
| `tt::version_patch`           | `constexpr int`  | `0`     |

The combined `TINYTENSOR_VERSION` macro encodes the version as `major * 10000 + minor * 100 + patch`.

---

### layout.hpp

**File:** `include/tinytensor/core/layout.hpp`

Defines the memory layout enum used to determine stride computation order.

```cpp
enum class layout_type { row_major, column_major };

inline constexpr layout_type default_layout = layout_type::row_major;
```

- **`row_major`** (C-style): The last dimension varies fastest in memory. This is the default.
- **`column_major`** (Fortran-style): The first dimension varies fastest in memory.

---

### shape.hpp

**File:** `include/tinytensor/core/shape.hpp`

#### `class shape_t`

A dynamically-sized vector of dimension sizes. Wraps `std::vector<std::size_t>` with convenience methods.

**Type aliases:**

| Alias            | Type                                  |
|------------------|---------------------------------------|
| `value_type`     | `std::size_t`                         |
| `container_type` | `std::vector<std::size_t>`            |
| `iterator`       | `container_type::iterator`            |
| `const_iterator` | `container_type::const_iterator`      |

**Constructors:**

| Signature                                      | Description                                  |
|------------------------------------------------|----------------------------------------------|
| `shape_t()`                                    | Default: empty shape (0 dimensions)          |
| `shape_t(std::initializer_list<value_type>)`   | Construct from dimension list, e.g. `{3,4,5}`|
| `shape_t(std::span<const value_type>)`         | Construct from a span (explicit)             |
| `shape_t(Iter first, Iter last)`               | Construct from iterator range                |

**Member functions:**

| Method              | Return type    | Description                                       |
|---------------------|----------------|---------------------------------------------------|
| `ndim()`            | `size_t`       | Number of dimensions                              |
| `size()`            | `size_t`       | Total number of elements (product of dimensions)  |
| `empty()`           | `bool`         | Whether the shape has zero dimensions             |
| `operator[](i)`     | `value_type&`  | Access dimension `i` (mutable and const)          |
| `data()`            | `value_type*`  | Pointer to underlying dimension array             |
| `begin()` / `end()` | iterator       | Range iteration over dimensions                   |
| `push_back(dim)`    | `void`         | Append a dimension                                |
| `resize(n)`         | `void`         | Resize the dimension count                        |
| `reserve(n)`        | `void`         | Reserve capacity                                  |
| `operator<=>()`     | auto           | Three-way comparison (defaulted)                  |

**Example:**

```cpp
tt::shape_t s{3, 4, 5};
s.ndim();  // 3
s.size();  // 60
s[0];      // 3
```

---

### strides.hpp

**File:** `include/tinytensor/core/strides.hpp`

Provides stride computation and index-to-offset mapping utilities.

**Type alias:**

```cpp
using strides_t = std::vector<std::ptrdiff_t>;
```

Strides are signed to support negative-stride views (reversed slicing).

#### `detail::compute_strides(shape, layout)`

Computes strides for a given shape and layout. Dimensions of size 1 receive stride 0 (broadcasting-ready).

```cpp
// Row-major strides for shape {3, 4, 5}: {20, 5, 1}
// Column-major strides for shape {3, 4, 5}: {1, 3, 12}
```

#### `detail::data_offset(strides, indices...)`

Computes the flat memory offset from a set of multi-dimensional indices. Two overloads:

- **Variadic template**: `data_offset(strides, idx0, idx1, ...)` -- compile-time index count.
- **Span-based**: `data_offset(strides, span<const size_t>)` -- runtime index count.

#### `detail::check_bounds(shape, indices...)`

Returns `true` if all indices are within the corresponding dimension bounds. Two overloads (variadic and span-based).

---

### concepts.hpp

**File:** `include/tinytensor/core/concepts.hpp`

Defines C++20 concepts for constraining template parameters.

| Concept              | Requirements                                             |
|----------------------|----------------------------------------------------------|
| `arithmetic<T>`      | `std::integral<T> \|\| std::floating_point<T>`           |
| `tensor_like<T>`     | Has `shape()`, `size()`, `data()`, `ndim()`              |
| `view_like<T>`       | `tensor_like<T>` + `offset()`, `strides()`               |
| `contiguous_tensor<T>`| `tensor_like<T>` + `is_contiguous()` returning `bool`   |

The `arithmetic` concept is used to constrain scalar operands in tensor-scalar arithmetic operators, preventing ambiguous overloads.

---

## Tensor Module

### tensor.hpp

**File:** `include/tinytensor/tensor/tensor.hpp`

#### `class tensor<T>`

The primary dynamic tensor class. Owns its data via `std::vector<T>`.

**Compile-time restriction:** `tensor<bool>` is explicitly disabled via `static_assert` because `std::vector<bool>` uses bit-packing, which breaks pointer semantics. Use `tensor<uint8_t>` instead.

**Type aliases:**

| Alias           | Type                                 |
|-----------------|--------------------------------------|
| `value_type`    | `T`                                  |
| `pointer`       | `T*`                                 |
| `reference`     | `T&`                                 |
| `size_type`     | `std::size_t`                        |
| `difference_type`| `std::ptrdiff_t`                    |
| `iterator`      | `std::vector<T>::iterator`           |
| `const_iterator`| `std::vector<T>::const_iterator`     |

**Constructors:**

| Signature                                                         | Description                                    |
|-------------------------------------------------------------------|------------------------------------------------|
| `tensor()`                                                        | Default: empty tensor                          |
| `tensor(const shape_t&, layout_type = default_layout)`            | Zero-initialized tensor with given shape       |
| `tensor(const shape_t&, const T& value, layout_type = ...)`      | Fill-initialized tensor                        |
| `tensor(initializer_list<T>)`                                     | 1D tensor from list                            |
| `tensor(initializer_list<initializer_list<T>>)`                   | 2D tensor from nested lists                    |
| `tensor(initializer_list<initializer_list<initializer_list<T>>>)` | 3D tensor from nested lists                    |
| `tensor(const tensor_view<T>&)` (explicit)                        | Materialize a view into an owning tensor       |

**Element access:**

| Method                        | Description                                          |
|-------------------------------|------------------------------------------------------|
| `operator()(indices...)`      | Variadic multi-dimensional index (checked in debug)  |
| `at(span<const size_type>)`   | Dynamic index via span                               |
| `at(vector<size_type>)`       | Dynamic index via vector                             |
| `flat(idx)`                   | Flat 1D index into underlying storage                |

All access methods perform bounds checking via `TT_ASSERT` (disabled in release builds with `NDEBUG`).

**Shape manipulation (in-place, return `*this`):**

| Method                   | Description                                                |
|--------------------------|------------------------------------------------------------|
| `reshape(shape_t)`       | Change shape, same total element count required            |
| `reshape({dims...})`     | Reshape with `-1` for one inferred dimension               |
| `resize(shape_t)`        | Change shape and size, may lose or gain (zero-init) data   |
| `resize(shape_t, fill)`  | Same as above, new elements initialized to `fill`          |
| `flatten()`              | Reshape to 1D                                              |
| `squeeze()`              | Remove all dimensions of size 1                            |
| `unsqueeze(axis)`        | Insert a dimension of size 1 at `axis`                     |

**Shape manipulation (return new tensor):**

| Method          | Description                      |
|-----------------|----------------------------------|
| `flattened()`   | Return a flattened copy          |
| `transposed()`  | Return a transposed copy         |
| `copy()`        | Return a deep copy               |

**Other methods:**

| Method                    | Description                                          |
|---------------------------|------------------------------------------------------|
| `view()`                  | Return a `tensor_view<T>` over the entire tensor     |
| `broadcast_to(shape_t)`   | Return a `broadcast_view<T>` to target shape         |
| `fill(value)`             | Set all elements to `value`                          |
| `copy_from(tensor<U>&)`   | Copy elements from another tensor (shapes must match)|
| `is_contiguous()`         | Always `true` for owning tensors                     |
| `ndim()`, `size()`, etc.  | Standard accessors                                   |

#### Free function: `view(tensor, slices...)`

Creates a `tensor_view` by applying slice specifications to a tensor or existing view.

```cpp
auto v = tt::view(t, tt::range(0, 2), tt::all);
auto v2 = tt::view(v, tt::range(1, 3));  // chained slicing
```

Three overloads exist: for `tensor<T>&`, `const tensor<T>&`, and `tensor_view<T>&`.

#### Factory functions

| Function                       | Signature                                       | Description                     |
|--------------------------------|-------------------------------------------------|---------------------------------|
| `zeros<T>(shape)`              | `tensor<T>`                                     | All elements zero               |
| `ones<T>(shape)`               | `tensor<T>`                                     | All elements one                |
| `full<T>(shape, value)`        | `tensor<T>`                                     | All elements set to `value`     |
| `arange<T>(start, stop, step)` | `tensor<T>`                                     | Range of values (1D)            |
| `linspace<T>(start, stop, n)`  | `tensor<T>`                                     | `n` evenly spaced values (1D)   |
| `eye<T>(n)`                    | `tensor<T>`                                     | n x n identity matrix           |
| `diag(tensor<T>)`              | `tensor<T>`                                     | Diagonal matrix from 1D tensor  |

---

### tensor_ops.hpp

**File:** `include/tinytensor/tensor/tensor_ops.hpp`

Provides element-wise operations on `tensor<T>`. All binary operations support NumPy-style broadcasting.

#### Internal helpers (namespace `detail`)

| Function                             | Description                                       |
|--------------------------------------|---------------------------------------------------|
| `binary_op(a, b, op)`               | Broadcast-aware tensor-tensor binary operation    |
| `binary_op_scalar(a, scalar, op)`   | Tensor-scalar binary operation                    |
| `scalar_binary_op(scalar, a, op)`   | Scalar-tensor binary operation                    |
| `unary_op(a, op)`                   | Element-wise unary operation                      |

#### Arithmetic operators

All return a new `tensor<result_type>`.

| Operator           | Variants                                 |
|--------------------|------------------------------------------|
| `+`                | tensor+tensor, tensor+scalar, scalar+tensor |
| `-`                | tensor-tensor, tensor-scalar, scalar-tensor |
| `*`                | tensor*tensor, tensor*scalar, scalar*tensor |
| `/`                | tensor/tensor, tensor/scalar, scalar/tensor |
| `-` (unary)        | Negation                                    |

Scalar operands are constrained by `arithmetic<S>`.

#### Math functions

All are element-wise and return a new tensor.

| Function      | Description           | Function      | Description           |
|---------------|-----------------------|---------------|-----------------------|
| `abs(a)`      | Absolute value        | `sin(a)`      | Sine                  |
| `sqrt(a)`     | Square root           | `cos(a)`      | Cosine                |
| `exp(a)`      | Exponential           | `tan(a)`      | Tangent               |
| `log(a)`      | Natural logarithm     | `asin(a)`     | Arcsine               |
| `log10(a)`    | Base-10 logarithm     | `acos(a)`     | Arccosine             |
| `log2(a)`     | Base-2 logarithm      | `atan(a)`     | Arctangent            |
| `floor(a)`    | Floor                 | `sinh(a)`     | Hyperbolic sine       |
| `ceil(a)`     | Ceiling               | `cosh(a)`     | Hyperbolic cosine     |
| `round(a)`    | Round                 | `tanh(a)`     | Hyperbolic tangent    |
| `pow(a, b)`   | Power (tensor or scalar exponent) |    |                       |

#### Reduction functions

| Function    | Return | Description                                     |
|-------------|--------|-------------------------------------------------|
| `sum(a)`    | `T`    | Sum of all elements                             |
| `prod(a)`   | `T`    | Product of all elements                         |
| `min(a)`    | `T`    | Minimum element                                 |
| `max(a)`    | `T`    | Maximum element                                 |
| `mean(a)`   | `T`    | Arithmetic mean                                 |

Note: These are global reductions only. Axis-wise reductions are not yet implemented.

#### Comparison operators

Element-wise comparisons return `tensor<uint8_t>` (not `tensor<bool>`, which is not supported).

| Operator | Description    |
|----------|----------------|
| `==`     | Equal          |
| `!=`     | Not equal      |
| `<`      | Less than      |
| `<=`     | Less or equal  |
| `>`      | Greater than   |
| `>=`     | Greater/equal  |

#### Boolean reductions

| Function       | Description                                    |
|----------------|------------------------------------------------|
| `all_of(a)`    | `true` if all elements of `tensor<uint8_t>` are non-zero |
| `any_of(a)`    | `true` if any element is non-zero              |

---

## Views Module

### slice.hpp

**File:** `include/tinytensor/views/slice.hpp`

Defines the slice specification types used by `tt::view()`.

#### Slice types

| Type          | Description                                          | Usage example              |
|---------------|------------------------------------------------------|----------------------------|
| `range_t`     | Start/stop/step range (like Python `slice`)          | `tt::range(0, 5, 2)`      |
| `all_tag`     | Select entire dimension                              | `tt::all`                  |
| `newaxis_tag` | Insert a new dimension of size 1                     | `tt::newaxis`              |
| `ellipsis_tag`| Expand to fill remaining dimensions                  | `tt::ellipsis`             |
| `ptrdiff_t`   | Integer index (collapses dimension)                  | `std::ptrdiff_t{2}`       |

The `slice_t` type is a variant:

```cpp
using slice_t = std::variant<std::ptrdiff_t, range_t, all_tag, newaxis_tag, ellipsis_tag>;
```

#### Constants

| Constant     | Type           | Value                                    |
|--------------|----------------|------------------------------------------|
| `tt::all`    | `all_tag`      | Select entire dimension                  |
| `tt::newaxis`| `newaxis_tag`  | Insert new axis                          |
| `tt::ellipsis`| `ellipsis_tag`| Expand remaining dims                    |
| `tt::_`      | `ptrdiff_t`    | Alias for `end_marker` (max ptrdiff_t)   |

#### `range(start, stop, step = 1)`

Creates a `range_t`. Supports negative indices (Python-style).

```cpp
tt::range(0, 5)       // elements 0..4
tt::range(1, tt::_, 2) // elements 1, 3, 5, ... to end
tt::range(-3, tt::_)   // last 3 elements
tt::range(4, -1, -1)   // elements 4, 3, 2, 1, 0 (reversed)
```

#### Index normalization (namespace `detail`)

- **`normalize_index(idx, size)`**: Converts negative indices to positive (`-1` -> `size - 1`).
- **`clamp_index(idx, size)`**: Clamps index to `[0, size]`.
- **`normalize_range(range_t, dim_size)`**: Resolves a `range_t` to a `normalized_range{start, size, step}`, handling negative indices, end markers, and clamping.

---

### tensor_view.hpp

**File:** `include/tinytensor/views/tensor_view.hpp`

#### `class tensor_view<T>`

A non-owning, strided view into tensor data. Supports arbitrary strides (including negative for reversed views) and non-contiguous memory access.

**Data members:**

| Member     | Type            | Description                              |
|------------|-----------------|------------------------------------------|
| `data_`    | `T*`            | Pointer to the underlying data buffer    |
| `offset_`  | `ptrdiff_t`     | Base offset into the data buffer         |
| `shape_`   | `shape_t`       | View shape                               |
| `strides_` | `strides_t`     | View strides (may differ from source)    |

**Element access:**

Same interface as `tensor<T>`: `operator()(indices...)` and `at(span)`.

The actual memory address for indices `(i, j, k, ...)` is:

```
data_[offset_ + strides_[0]*i + strides_[1]*j + strides_[2]*k + ...]
```

**`is_contiguous()`**: Returns `true` if the view represents a contiguous memory region (strides match what `compute_strides` would produce for the view's shape).

**Iteration:** Provides `iterator` and `const_iterator` (forward iterators) that traverse the view in logical order, correctly handling non-contiguous strides.

#### `detail::apply_slices(data, offset, shape, strides, slices)`

The core slicing engine. Applies a sequence of `slice_t` values to produce a new `tensor_view<T>` with adjusted offset, shape, and strides.

Slice processing rules:

| Slice type    | Effect on output                                            |
|---------------|-------------------------------------------------------------|
| Integer       | Collapses that dimension (reduces ndim by 1), adjusts offset|
| `range_t`     | Selects a subrange, adjusts offset/stride/shape             |
| `all_tag`     | Keeps the dimension unchanged                               |
| `newaxis_tag` | Inserts a new dimension with size 1 and stride 0            |
| `ellipsis_tag`| Expands to cover remaining unaddressed dimensions           |

Constraints enforced:
- At most one `ellipsis` is allowed.
- The number of non-newaxis slice entries must not exceed `shape.ndim()` (unless an ellipsis is present).

---

### broadcast.hpp

**File:** `include/tinytensor/views/broadcast.hpp`

#### `class broadcast_view<T>`

A read-only view that presents a tensor as if it had a larger shape, by repeating elements along broadcast dimensions (stride = 0).

Structurally similar to `tensor_view<T>` but provides only const access. The `operator()` is const-only. Provides a forward `iterator`.

#### `detail::broadcast_shapes(a, b, result)`

Computes the broadcast-compatible shape of two shapes following NumPy broadcasting rules:

1. Shapes are right-aligned.
2. Dimensions are compatible if they are equal, or one of them is 1.
3. The output dimension is the maximum of the two.

Returns `false` if shapes are incompatible.

#### `detail::broadcast_strides(original_shape, original_strides, broadcast_shape)`

Computes new strides for broadcasting `original_shape` to `broadcast_shape`:

- Dimensions that match keep their original stride.
- Dimensions of size 1 that are broadcast receive stride 0.
- Throws `broadcast_error` on incompatible shapes or lower-rank targets.

#### `broadcast(data, offset, original_shape, original_strides, target_shape)`

Free function that creates a `broadcast_view<T>`.

---

## Utilities Module

### assert.hpp

**File:** `include/tinytensor/utils/assert.hpp`

#### Exception hierarchy

```
std::runtime_error
└── tt::tensor_error           # Base for all tinytensor errors
    ├── tt::shape_error        # Shape mismatch errors
    ├── tt::index_error        # Index out-of-bounds errors
    └── tt::broadcast_error    # Incompatible broadcast shapes
```

#### Macros

| Macro                          | Behavior                                                  |
|--------------------------------|-----------------------------------------------------------|
| `TT_ASSERT(cond, msg)`        | Throws `tensor_error` with message + file:line if `cond` is false. Disabled (`((void)0)`) when `NDEBUG` is defined. |
| `TT_THROW(ExceptionType, msg)`| Unconditionally throws the specified exception type.      |

---

## API Reference

### Quick usage examples

#### Creating tensors

```cpp
// From shape
tt::tensor<double> a(tt::shape_t{3, 4, 5});

// From initializer lists (1D, 2D, 3D)
tt::tensor<int> v = {1, 2, 3, 4, 5};
tt::tensor<int> m = {{1, 2, 3}, {4, 5, 6}};
tt::tensor<int> t = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};

// Factory functions
auto z = tt::zeros<double>(tt::shape_t{3, 3});
auto o = tt::ones<float>(tt::shape_t{2, 4});
auto f = tt::full<int>(tt::shape_t{5}, 42);
auto r = tt::arange<int>(0, 10, 2);     // {0, 2, 4, 6, 8}
auto l = tt::linspace<double>(0, 1, 5); // {0, 0.25, 0.5, 0.75, 1.0}
auto I = tt::eye<double>(3);            // 3x3 identity
```

#### Element access

```cpp
tt::tensor<int> m = {{1, 2, 3}, {4, 5, 6}};

int val = m(0, 2);            // 3 (variadic)
m(1, 0) = 99;                 // modify

std::vector<size_t> idx{1, 2};
int val2 = m.at(idx);         // 6 (dynamic)

int val3 = m.flat(3);         // 4 (flat index)
```

#### Slicing and views

```cpp
tt::tensor<int> t = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// Subrange
auto v1 = tt::view(t, tt::range(2, 7));       // {3, 4, 5, 6, 7}

// Step
auto v2 = tt::view(t, tt::range(0, tt::_, 2));// {1, 3, 5, 7, 9}

// Negative indices
auto v3 = tt::view(t, tt::range(-3, tt::_));   // {8, 9, 10}

// Reverse
auto v4 = tt::view(t, tt::range(9, -1, -1));  // {10, 9, ..., 1}

// 2D slicing
tt::tensor<int> m = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
auto sub = tt::view(m, tt::range(0, 2), tt::range(1, 3));  // 2x2 submatrix

// Integer index (reduces dimension)
auto row = tt::view(m, std::ptrdiff_t{1}, tt::all);  // 1D: {5, 6, 7, 8}

// newaxis (adds dimension)
auto col = tt::view(t, tt::newaxis, tt::all);  // shape {1, 10}

// ellipsis
auto e = tt::view(m, tt::ellipsis, std::ptrdiff_t{0});  // first column

// Chained views
auto v5 = tt::view(m, tt::range(0, 2), tt::all);
auto v6 = tt::view(v5, tt::all, tt::range(1, 3));  // still references m
```

Views are non-owning. Mutations through a view modify the underlying tensor.

#### Broadcasting

```cpp
tt::tensor<double> a = {{1, 2, 3}, {4, 5, 6}};  // shape {2, 3}
tt::tensor<double> b = {10, 20, 30};              // shape {3}

auto result = a + b;  // broadcasts b to {2, 3}
// result: {{11, 22, 33}, {14, 25, 36}}
```

#### Math operations

```cpp
tt::tensor<double> x = {1.0, 4.0, 9.0};
auto s = tt::sqrt(x);   // {1, 2, 3}
auto e = tt::exp(x);
auto l = tt::log(x);
auto p = tt::pow(x, 0.5);

double total = tt::sum(x);   // 14.0
double avg = tt::mean(x);    // ~4.67
double mx = tt::max(x);      // 9.0
```

---

## Design Decisions

### 1. `tensor<bool>` is disabled

`std::vector<bool>` is a specialized container that uses bit-packing. This breaks `data()` (no `bool*` pointer) and prevents interoperability with C APIs and pointer-based views. The library explicitly `static_assert`s against `tensor<bool>` and uses `tensor<uint8_t>` for boolean-like data.

### 2. Strides use `ptrdiff_t`

Strides are signed (`std::ptrdiff_t`) rather than unsigned to support negative-stride views (e.g., `tt::range(4, -1, -1)` for reversed traversal). The offset in `tensor_view` is also `ptrdiff_t`.

### 3. Broadcast dimensions have stride 0

When computing strides, dimensions of size 1 receive stride 0. This means accessing any index along that dimension reads the same memory location, implementing broadcasting without copying data.

### 4. Assertions are debug-only

`TT_ASSERT` is a no-op when `NDEBUG` is defined, following the convention of standard `assert()`. This allows bounds-checked debug builds with zero overhead in release builds. `TT_THROW` is always active for errors that cannot be silently ignored (e.g., broadcast incompatibility).

### 5. Header-only design

All code resides in `.hpp` files. There are no compiled `.cpp` sources (outside of tests). This simplifies integration -- users only need to add the `include` directory to their include path.

### 6. Row-major default

The default memory layout is row-major (C-style), consistent with NumPy, xtensor, and most C++ scientific computing libraries. Column-major (Fortran-style) is supported but must be explicitly requested.

### 7. No lazy evaluation

Unlike xtensor, tinytensor evaluates operations eagerly. Every arithmetic operator and math function immediately allocates and computes its result tensor. This trades some performance for simplicity and predictability.

### 8. Views do not own data

`tensor_view<T>` and `broadcast_view<T>` hold raw pointers to the underlying tensor's data. The user must ensure the source tensor outlives any views derived from it. Materializing a view into an owning tensor is done via `tt::tensor<T>(view)`.
