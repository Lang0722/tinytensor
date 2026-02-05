// Define _USE_MATH_DEFINES before including cmath for M_PI on MSVC
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <cmath>

#include "../include/tinytensor/tinytensor.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// =============================================================================
// Shape Tests
// =============================================================================

TEST(ShapeTest, DefaultConstruction) {
  tt::shape_t s;
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.ndim(), 0);
  EXPECT_EQ(s.size(), 0);
}

TEST(ShapeTest, InitializerListConstruction) {
  tt::shape_t s{3, 4, 5};
  EXPECT_EQ(s.ndim(), 3);
  EXPECT_EQ(s.size(), 60);
  EXPECT_EQ(s[0], 3);
  EXPECT_EQ(s[1], 4);
  EXPECT_EQ(s[2], 5);
}

TEST(ShapeTest, Comparison) {
  tt::shape_t s1{3, 4, 5};
  tt::shape_t s2{3, 4, 5};
  tt::shape_t s3{3, 4, 6};

  EXPECT_EQ(s1, s2);
  EXPECT_NE(s1, s3);
  EXPECT_LT(s1, s3);
}

TEST(ShapeTest, Iteration) {
  tt::shape_t s{2, 3, 4};
  std::vector<std::size_t> expected{2, 3, 4};
  std::vector<std::size_t> actual(s.begin(), s.end());
  EXPECT_EQ(actual, expected);
}

TEST(ShapeTest, Modification) {
  tt::shape_t s{2, 3};
  s.push_back(4);
  EXPECT_EQ(s.ndim(), 3);
  EXPECT_EQ(s[2], 4);

  s[0] = 5;
  EXPECT_EQ(s[0], 5);
}

// =============================================================================
// Strides Tests
// =============================================================================

TEST(StridesTest, RowMajor) {
  tt::shape_t s{3, 4, 5};
  auto strides = tt::detail::compute_strides(s, tt::layout_type::row_major);

  EXPECT_EQ(strides.size(), 3);
  EXPECT_EQ(strides[0], 20);  // 4 * 5
  EXPECT_EQ(strides[1], 5);   // 5
  EXPECT_EQ(strides[2], 1);   // 1
}

TEST(StridesTest, ColumnMajor) {
  tt::shape_t s{3, 4, 5};
  auto strides = tt::detail::compute_strides(s, tt::layout_type::column_major);

  EXPECT_EQ(strides.size(), 3);
  EXPECT_EQ(strides[0], 1);   // 1
  EXPECT_EQ(strides[1], 3);   // 3
  EXPECT_EQ(strides[2], 12);  // 3 * 4
}

TEST(StridesTest, BroadcastStrides) {
  tt::shape_t s{3, 1, 5};
  auto strides = tt::detail::compute_strides(s, tt::layout_type::row_major);
  EXPECT_EQ(strides[1], 0);  // broadcast dimension
}

TEST(StridesTest, DataOffset) {
  tt::strides_t strides = {20, 5, 1};

  EXPECT_EQ(tt::detail::data_offset(strides, 0, 0, 0), 0);
  EXPECT_EQ(tt::detail::data_offset(strides, 1, 2, 3), 1 * 20 + 2 * 5 + 3);
  EXPECT_EQ(tt::detail::data_offset(strides, 2, 3, 4), 2 * 20 + 3 * 5 + 4);
}

TEST(StridesTest, EmptyShape) {
  tt::shape_t s;
  auto strides = tt::detail::compute_strides(s);
  EXPECT_TRUE(strides.empty());
}

// =============================================================================
// Tensor Construction Tests
// =============================================================================

TEST(TensorTest, ConstructFromShape) {
  tt::tensor<double> t(tt::shape_t{3, 4, 5});
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(t.size(), 60);
  EXPECT_EQ(t.shape()[0], 3);
  EXPECT_EQ(t.shape()[1], 4);
  EXPECT_EQ(t.shape()[2], 5);
}

TEST(TensorTest, ConstructFromShapeWithValue) {
  tt::tensor<int> t(tt::shape_t{2, 3}, 42);
  EXPECT_EQ(t.size(), 6);
  for (const auto& val : t) {
    EXPECT_EQ(val, 42);
  }
}

TEST(TensorTest, ConstructFrom1DInitializerList) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};
  EXPECT_EQ(t.ndim(), 1);
  EXPECT_EQ(t.size(), 5);
  EXPECT_EQ(t(0), 1);
  EXPECT_EQ(t(4), 5);
}

TEST(TensorTest, ConstructFrom2DInitializerList) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(t.shape()[0], 2);
  EXPECT_EQ(t.shape()[1], 3);
  EXPECT_EQ(t(0, 0), 1);
  EXPECT_EQ(t(1, 2), 6);
}

TEST(TensorTest, ConstructFrom3DInitializerList) {
  tt::tensor<int> t = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(t.shape()[0], 2);
  EXPECT_EQ(t.shape()[1], 2);
  EXPECT_EQ(t.shape()[2], 2);
  EXPECT_EQ(t(0, 0, 0), 1);
  EXPECT_EQ(t(1, 1, 1), 8);
}

// =============================================================================
// Tensor Access Tests
// =============================================================================

TEST(TensorTest, ElementAccess) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};

  EXPECT_EQ(t(0, 0), 1);
  EXPECT_EQ(t(0, 1), 2);
  EXPECT_EQ(t(1, 0), 4);
  EXPECT_EQ(t(1, 2), 6);

  t(0, 1) = 99;
  EXPECT_EQ(t(0, 1), 99);
}

TEST(TensorTest, DynamicIndexAccess) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};

  std::vector<std::size_t> idx1 = {0, 1};
  EXPECT_EQ(t.at(idx1), 2);

  std::vector<std::size_t> idx2 = {1, 2};
  t.at(idx2) = 100;
  EXPECT_EQ(t(1, 2), 100);
}

TEST(TensorTest, FlatAccess) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};

  EXPECT_EQ(t.flat(0), 1);
  EXPECT_EQ(t.flat(4), 5);

  t.flat(2) = 99;
  EXPECT_EQ(t(2), 99);
}

TEST(TensorTest, Iteration) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};

  int sum = 0;
  for (const auto& val : t) {
    sum += val;
  }
  EXPECT_EQ(sum, 15);
}

// =============================================================================
// Tensor Reshape Tests
// =============================================================================

TEST(TensorTest, ReshapeSameSize) {
  tt::tensor<int> t = {1, 2, 3, 4, 5, 6};
  t.reshape(tt::shape_t{2, 3});

  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(t.shape()[0], 2);
  EXPECT_EQ(t.shape()[1], 3);
  EXPECT_EQ(t(0, 0), 1);
  EXPECT_EQ(t(1, 2), 6);
}

TEST(TensorTest, ReshapeWithInference) {
  tt::tensor<int> t = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  t.reshape({3, -1});
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(t.shape()[0], 3);
  EXPECT_EQ(t.shape()[1], 4);

  t.reshape({-1, 2, 2});
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(t.shape()[0], 3);
  EXPECT_EQ(t.shape()[1], 2);
  EXPECT_EQ(t.shape()[2], 2);
}

TEST(TensorTest, ReshapeToDifferentDimensions) {
  tt::tensor<int> t(tt::shape_t{24});
  for (std::size_t i = 0; i < 24; ++i) {
    t(i) = static_cast<int>(i);
  }

  // 1D -> 2D
  t.reshape(tt::shape_t{4, 6});
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(t(0, 0), 0);
  EXPECT_EQ(t(3, 5), 23);

  // 2D -> 3D
  t.reshape(tt::shape_t{2, 3, 4});
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(t(0, 0, 0), 0);
  EXPECT_EQ(t(1, 2, 3), 23);

  // 3D -> 4D
  t.reshape(tt::shape_t{2, 2, 2, 3});
  EXPECT_EQ(t.ndim(), 4);
  EXPECT_EQ(t(0, 0, 0, 0), 0);
  EXPECT_EQ(t(1, 1, 1, 2), 23);

  // 4D -> 1D
  t.flatten();
  EXPECT_EQ(t.ndim(), 1);
  EXPECT_EQ(t(0), 0);
  EXPECT_EQ(t(23), 23);
}

TEST(TensorTest, Resize) {
  tt::tensor<int> t = {1, 2, 3};

  t.resize(tt::shape_t{5}, 0);
  EXPECT_EQ(t.size(), 5);
  EXPECT_EQ(t(0), 1);
  EXPECT_EQ(t(1), 2);
  EXPECT_EQ(t(2), 3);
  EXPECT_EQ(t(3), 0);
  EXPECT_EQ(t(4), 0);

  t.resize(tt::shape_t{2, 2});
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(t.size(), 4);
}

TEST(TensorTest, Flatten) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};
  t.flatten();

  EXPECT_EQ(t.ndim(), 1);
  EXPECT_EQ(t.size(), 6);
  EXPECT_EQ(t(0), 1);
  EXPECT_EQ(t(5), 6);
}

TEST(TensorTest, Squeeze) {
  tt::tensor<int> t(tt::shape_t{1, 3, 1, 4, 1}, 0);
  t.squeeze();

  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(t.shape()[0], 3);
  EXPECT_EQ(t.shape()[1], 4);
}

TEST(TensorTest, Unsqueeze) {
  tt::tensor<int> t = {1, 2, 3};

  t.unsqueeze(0);
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(t.shape()[0], 1);
  EXPECT_EQ(t.shape()[1], 3);

  t.unsqueeze(2);
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(t.shape()[2], 1);
}

TEST(TensorTest, Transpose) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};
  auto tr = t.transposed();

  EXPECT_EQ(tr.ndim(), 2);
  EXPECT_EQ(tr.shape()[0], 3);
  EXPECT_EQ(tr.shape()[1], 2);
  EXPECT_EQ(tr(0, 0), 1);
  EXPECT_EQ(tr(0, 1), 4);
  EXPECT_EQ(tr(2, 1), 6);
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(TensorTest, Zeros) {
  auto z = tt::zeros<double>(tt::shape_t{2, 3});
  EXPECT_EQ(z.size(), 6);
  for (const auto& val : z) {
    EXPECT_DOUBLE_EQ(val, 0.0);
  }
}

TEST(TensorTest, Ones) {
  auto o = tt::ones<int>(tt::shape_t{3, 3});
  EXPECT_EQ(o.size(), 9);
  for (const auto& val : o) {
    EXPECT_EQ(val, 1);
  }
}

TEST(TensorTest, Full) {
  auto f = tt::full<float>(tt::shape_t{2, 2}, 3.14f);
  for (const auto& val : f) {
    EXPECT_FLOAT_EQ(val, 3.14f);
  }
}

TEST(TensorTest, Arange) {
  auto a = tt::arange<int>(0, 5);
  EXPECT_EQ(a.size(), 5);
  EXPECT_EQ(a(0), 0);
  EXPECT_EQ(a(4), 4);

  auto a2 = tt::arange<int>(2, 10, 2);
  EXPECT_EQ(a2.size(), 4);
  EXPECT_EQ(a2(0), 2);
  EXPECT_EQ(a2(3), 8);
}

TEST(TensorTest, Linspace) {
  auto l = tt::linspace<double>(0.0, 1.0, 5);
  EXPECT_EQ(l.size(), 5);
  EXPECT_NEAR(l(0), 0.0, 1e-10);
  EXPECT_NEAR(l(4), 1.0, 1e-10);
}

TEST(TensorTest, Eye) {
  auto e = tt::eye<double>(3);
  EXPECT_EQ(e.ndim(), 2);
  EXPECT_EQ(e.shape()[0], 3);
  EXPECT_EQ(e.shape()[1], 3);
  EXPECT_DOUBLE_EQ(e(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(e(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(e(2, 2), 1.0);
  EXPECT_DOUBLE_EQ(e(0, 1), 0.0);
}

TEST(TensorTest, Diag) {
  tt::tensor<int> v = {1, 2, 3};
  auto d = tt::diag(v);

  EXPECT_EQ(d.ndim(), 2);
  EXPECT_EQ(d.shape()[0], 3);
  EXPECT_EQ(d(0, 0), 1);
  EXPECT_EQ(d(1, 1), 2);
  EXPECT_EQ(d(2, 2), 3);
  EXPECT_EQ(d(0, 1), 0);
}

// =============================================================================
// Slice Tests
// =============================================================================

TEST(SliceTest, RangeCreation) {
  auto r1 = tt::range(0, 5);
  EXPECT_EQ(r1.start, 0);
  EXPECT_EQ(r1.stop, 5);
  EXPECT_EQ(r1.step, 1);

  auto r2 = tt::range(1, 10, 2);
  EXPECT_EQ(r2.start, 1);
  EXPECT_EQ(r2.stop, 10);
  EXPECT_EQ(r2.step, 2);
}

TEST(SliceTest, NormalizeIndex) {
  EXPECT_EQ(tt::detail::normalize_index(0, 10), 0);
  EXPECT_EQ(tt::detail::normalize_index(5, 10), 5);
  EXPECT_EQ(tt::detail::normalize_index(-1, 10), 9);
  EXPECT_EQ(tt::detail::normalize_index(-10, 10), 0);
}

TEST(SliceTest, NormalizeRange) {
  auto nr1 = tt::detail::normalize_range(tt::range(0, 5), 10);
  EXPECT_EQ(nr1.start, 0);
  EXPECT_EQ(nr1.size, 5);

  auto nr2 = tt::detail::normalize_range(tt::range(-3, tt::_), 10);
  EXPECT_EQ(nr2.start, 7);
  EXPECT_EQ(nr2.size, 3);

  auto nr3 = tt::detail::normalize_range(tt::range(0, 10, 2), 10);
  EXPECT_EQ(nr3.size, 5);

  // range(9, -1, -1) with dim=10: stop=-1 normalizes to 9, start==stop → empty
  auto nr4 = tt::detail::normalize_range(tt::range(9, -1, -1), 10);
  EXPECT_EQ(nr4.start, 9);
  EXPECT_EQ(nr4.size, 0);

  // Use tt::_ (end_marker) to reverse all the way to index 0
  auto nr5 = tt::detail::normalize_range(tt::range(9, tt::_, -1), 10);
  EXPECT_EQ(nr5.start, 9);
  EXPECT_EQ(nr5.size, 10);
}

// =============================================================================
// View Tests
// =============================================================================

TEST(ViewTest, BasicView) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  auto v = t.view();
  EXPECT_EQ(v.ndim(), 2);
  EXPECT_EQ(v.shape()[0], 3);
  EXPECT_EQ(v(0, 0), 1);
  EXPECT_EQ(v(2, 2), 9);

  v(1, 1) = 99;
  EXPECT_EQ(t(1, 1), 99);
}

TEST(ViewTest, SliceInteger) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  auto v = tt::view(t, std::ptrdiff_t{1}, tt::all);
  EXPECT_EQ(v.ndim(), 1);
  EXPECT_EQ(v.shape()[0], 3);
  EXPECT_EQ(v(0), 4);
  EXPECT_EQ(v(1), 5);
  EXPECT_EQ(v(2), 6);
}

TEST(ViewTest, SliceRange) {
  tt::tensor<int> t = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto v1 = tt::view(t, tt::range(2, 7));
  EXPECT_EQ(v1.shape()[0], 5);
  EXPECT_EQ(v1(0), 3);
  EXPECT_EQ(v1(4), 7);

  auto v2 = tt::view(t, tt::range(0, tt::_, 2));
  EXPECT_EQ(v2.shape()[0], 5);
  EXPECT_EQ(v2(0), 1);
  EXPECT_EQ(v2(1), 3);

  auto v3 = tt::view(t, tt::range(-3, tt::_));
  EXPECT_EQ(v3.shape()[0], 3);
  EXPECT_EQ(v3(0), 8);
}

TEST(ViewTest, Slice2D) {
  tt::tensor<int> t = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};

  auto v = tt::view(t, tt::range(0, 2), tt::range(1, 3));
  EXPECT_EQ(v.ndim(), 2);
  EXPECT_EQ(v.shape()[0], 2);
  EXPECT_EQ(v.shape()[1], 2);
  EXPECT_EQ(v(0, 0), 2);
  EXPECT_EQ(v(1, 1), 7);
}

TEST(ViewTest, ViewMutation) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};

  auto v = tt::view(t, tt::range(0, 2), tt::range(1, 3));
  v(0, 0) = 100;
  v(1, 1) = 200;

  EXPECT_EQ(t(0, 1), 100);
  EXPECT_EQ(t(1, 2), 200);
}

TEST(ViewTest, Newaxis) {
  tt::tensor<int> t = {1, 2, 3};

  auto v = tt::view(t, tt::newaxis, tt::all);
  EXPECT_EQ(v.ndim(), 2);
  EXPECT_EQ(v.shape()[0], 1);
  EXPECT_EQ(v.shape()[1], 3);
  EXPECT_EQ(v(0, 0), 1);
}

TEST(ViewTest, Ellipsis) {
  tt::tensor<int> t(tt::shape_t{2, 3, 4});
  int val = 0;
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 4; ++k) {
        t(i, j, k) = val++;
      }
    }
  }

  auto v = tt::view(t, tt::ellipsis, std::ptrdiff_t{0});
  EXPECT_EQ(v.ndim(), 2);
  EXPECT_EQ(v.shape()[0], 2);
  EXPECT_EQ(v.shape()[1], 3);
  EXPECT_EQ(v(0, 0), 0);
  EXPECT_EQ(v(1, 2), 20);
}

TEST(ViewTest, ViewIteration) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};
  auto v = tt::view(t, tt::range(1, 4));

  int sum = 0;
  for (const auto& val : v) {
    sum += val;
  }
  EXPECT_EQ(sum, 2 + 3 + 4);
}

TEST(ViewTest, IsContiguous) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};

  auto v1 = t.view();
  EXPECT_TRUE(v1.is_contiguous());

  auto v2 = tt::view(t, tt::all, tt::range(0, 2));
  EXPECT_FALSE(v2.is_contiguous());
}

TEST(ViewTest, ViewChains) {
  tt::tensor<int> t = {{1, 2, 3, 4, 5},
                       {6, 7, 8, 9, 10},
                       {11, 12, 13, 14, 15},
                       {16, 17, 18, 19, 20}};

  auto v1 = tt::view(t, tt::range(1, 3), tt::all);
  auto v2 = tt::view(v1, tt::all, tt::range(2, 4));

  EXPECT_EQ(v2(0, 0), 8);
  EXPECT_EQ(v2(1, 1), 14);

  v2(0, 0) = 999;
  EXPECT_EQ(t(1, 2), 999);
}

// =============================================================================
// Broadcast Tests
// =============================================================================

TEST(BroadcastTest, BroadcastShapes) {
  tt::shape_t a{3, 4, 5};
  tt::shape_t b{5};
  tt::shape_t result;

  EXPECT_TRUE(tt::detail::broadcast_shapes(a, b, result));
  EXPECT_EQ(result.ndim(), 3);
  EXPECT_EQ(result[0], 3);
  EXPECT_EQ(result[1], 4);
  EXPECT_EQ(result[2], 5);
}

TEST(BroadcastTest, BroadcastShapesExpand) {
  tt::shape_t a{1, 4, 1};
  tt::shape_t b{3, 1, 5};
  tt::shape_t result;

  EXPECT_TRUE(tt::detail::broadcast_shapes(a, b, result));
  EXPECT_EQ(result[0], 3);
  EXPECT_EQ(result[1], 4);
  EXPECT_EQ(result[2], 5);
}

TEST(BroadcastTest, BroadcastShapesIncompatible) {
  tt::shape_t a{3, 4};
  tt::shape_t b{5, 6};
  tt::shape_t result;

  EXPECT_FALSE(tt::detail::broadcast_shapes(a, b, result));
}

TEST(BroadcastTest, BroadcastView) {
  tt::tensor<int> t = {1, 2, 3};
  auto bc = t.broadcast_to(tt::shape_t{2, 3});

  EXPECT_EQ(bc.ndim(), 2);
  EXPECT_EQ(bc.shape()[0], 2);
  EXPECT_EQ(bc.shape()[1], 3);
  EXPECT_EQ(bc(0, 0), 1);
  EXPECT_EQ(bc(1, 2), 3);
}

TEST(BroadcastTest, BroadcastIteration) {
  tt::tensor<int> t = {10, 20};
  auto bc = t.broadcast_to(tt::shape_t{3, 2});

  int sum = 0;
  for (const auto& val : bc) {
    sum += val;
  }
  EXPECT_EQ(sum, 3 * (10 + 20));
}

// =============================================================================
// Operations Tests
// =============================================================================

TEST(OpsTest, ArithmeticTensorTensor) {
  tt::tensor<double> a = {1.0, 2.0, 3.0};
  tt::tensor<double> b = {4.0, 5.0, 6.0};

  auto add = a + b;
  EXPECT_DOUBLE_EQ(add(0), 5.0);
  EXPECT_DOUBLE_EQ(add(1), 7.0);
  EXPECT_DOUBLE_EQ(add(2), 9.0);

  auto sub = a - b;
  EXPECT_DOUBLE_EQ(sub(0), -3.0);

  auto mul = a * b;
  EXPECT_DOUBLE_EQ(mul(0), 4.0);
  EXPECT_DOUBLE_EQ(mul(1), 10.0);

  auto div = b / a;
  EXPECT_DOUBLE_EQ(div(0), 4.0);
  EXPECT_DOUBLE_EQ(div(1), 2.5);
}

TEST(OpsTest, ArithmeticTensorScalar) {
  tt::tensor<double> a = {1.0, 2.0, 3.0};

  auto add = a + 10.0;
  EXPECT_DOUBLE_EQ(add(0), 11.0);

  auto mul = a * 2.0;
  EXPECT_DOUBLE_EQ(mul(0), 2.0);
  EXPECT_DOUBLE_EQ(mul(1), 4.0);
}

TEST(OpsTest, ArithmeticScalarTensor) {
  tt::tensor<double> a = {1.0, 2.0, 4.0};

  auto sub = 10.0 - a;
  EXPECT_DOUBLE_EQ(sub(0), 9.0);
  EXPECT_DOUBLE_EQ(sub(1), 8.0);

  auto div = 8.0 / a;
  EXPECT_DOUBLE_EQ(div(0), 8.0);
  EXPECT_DOUBLE_EQ(div(1), 4.0);
}

TEST(OpsTest, UnaryOps) {
  tt::tensor<double> a = {-1.0, 2.0, -3.0};

  auto neg = -a;
  EXPECT_DOUBLE_EQ(neg(0), 1.0);
  EXPECT_DOUBLE_EQ(neg(1), -2.0);

  auto abs_a = tt::abs(a);
  EXPECT_DOUBLE_EQ(abs_a(0), 1.0);
  EXPECT_DOUBLE_EQ(abs_a(2), 3.0);
}

TEST(OpsTest, MathFunctions) {
  tt::tensor<double> a = {1.0, 4.0, 9.0};

  auto s = tt::sqrt(a);
  EXPECT_DOUBLE_EQ(s(0), 1.0);
  EXPECT_DOUBLE_EQ(s(1), 2.0);
  EXPECT_DOUBLE_EQ(s(2), 3.0);

  tt::tensor<double> b = {0.0, 1.0};
  auto e = tt::exp(b);
  EXPECT_NEAR(e(0), 1.0, 1e-10);
  EXPECT_NEAR(e(1), std::exp(1.0), 1e-10);
}

TEST(OpsTest, TrigFunctions) {
  tt::tensor<double> a = {0.0, M_PI / 2, M_PI};

  auto s = tt::sin(a);
  EXPECT_NEAR(s(0), 0.0, 1e-10);
  EXPECT_NEAR(s(1), 1.0, 1e-10);
  EXPECT_NEAR(s(2), 0.0, 1e-10);

  auto c = tt::cos(a);
  EXPECT_NEAR(c(0), 1.0, 1e-10);
  EXPECT_NEAR(c(1), 0.0, 1e-10);
  EXPECT_NEAR(c(2), -1.0, 1e-10);
}

TEST(OpsTest, Pow) {
  tt::tensor<double> base = {2.0, 3.0, 4.0};

  auto p = tt::pow(base, 2.0);
  EXPECT_DOUBLE_EQ(p(0), 4.0);
  EXPECT_DOUBLE_EQ(p(1), 9.0);
  EXPECT_DOUBLE_EQ(p(2), 16.0);
}

TEST(OpsTest, Reductions) {
  tt::tensor<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};

  EXPECT_DOUBLE_EQ(tt::sum(a), 15.0);
  EXPECT_DOUBLE_EQ(tt::prod(a), 120.0);
  EXPECT_DOUBLE_EQ(tt::min(a), 1.0);
  EXPECT_DOUBLE_EQ(tt::max(a), 5.0);
  EXPECT_DOUBLE_EQ(tt::mean(a), 3.0);
}

TEST(OpsTest, ComparisonOps) {
  tt::tensor<int> a = {1, 2, 3};
  tt::tensor<int> b = {1, 3, 2};

  auto eq = (a == b);
  auto eq_it = eq.begin();
  EXPECT_TRUE(*eq_it++);
  EXPECT_FALSE(*eq_it++);
  EXPECT_FALSE(*eq_it++);

  auto lt = (a < b);
  auto lt_it = lt.begin();
  EXPECT_FALSE(*lt_it++);
  EXPECT_TRUE(*lt_it++);
  EXPECT_FALSE(*lt_it++);
}

TEST(OpsTest, AllAny) {
  tt::tensor<uint8_t> all_true = {1, 1, 1};
  tt::tensor<uint8_t> some_true = {1, 0, 1};
  tt::tensor<uint8_t> all_false = {0, 0, 0};

  EXPECT_TRUE(tt::all_of(all_true));
  EXPECT_FALSE(tt::all_of(some_true));

  EXPECT_TRUE(tt::any_of(all_true));
  EXPECT_TRUE(tt::any_of(some_true));
  EXPECT_FALSE(tt::any_of(all_false));
}

TEST(OpsTest, BroadcastArithmetic) {
  tt::tensor<double> a = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  tt::tensor<double> b = {10.0, 20.0, 30.0};

  auto result = a + b;
  EXPECT_EQ(result.shape()[0], 2);
  EXPECT_EQ(result.shape()[1], 3);
  EXPECT_DOUBLE_EQ(result(0, 0), 11.0);
  EXPECT_DOUBLE_EQ(result(1, 2), 36.0);
}

// =============================================================================
// Fixed Tensor Tests
// =============================================================================

TEST(FixedTensorTest, Construction) {
  tt::fixed_tensor<int, 2, 3> t1;
  EXPECT_EQ(t1.ndim(), 2);
  EXPECT_EQ(t1.size(), 6);

  tt::fixed_tensor<double, 3, 3> t2(1.5);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(t2(i, j), 1.5);
    }
  }
}

TEST(FixedTensorTest, Access) {
  tt::fixed_tensor<int, 2, 3> t;
  t(0, 0) = 1;
  t(1, 2) = 6;

  EXPECT_EQ(t(0, 0), 1);
  EXPECT_EQ(t(1, 2), 6);
}

TEST(FixedTensorTest, CompileTimeProperties) {
  using tensor_type = tt::fixed_tensor<double, 3, 4, 5>;

  static_assert(tensor_type::ndim_v == 3);
  static_assert(tensor_type::size_v == 60);
  static_assert(tensor_type::shape_v[0] == 3);
  static_assert(tensor_type::strides_v[0] == 20);
}

TEST(FixedTensorTest, FactoryFunctions) {
  auto z = tt::zeros_fixed<double, 2, 3>();
  for (const auto& val : z) {
    EXPECT_DOUBLE_EQ(val, 0.0);
  }

  auto o = tt::ones_fixed<int, 3, 3>();
  for (const auto& val : o) {
    EXPECT_EQ(val, 1);
  }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST(EdgeCaseTest, EmptyTensor) {
  tt::shape_t empty;
  tt::tensor<int> t(empty);
  EXPECT_TRUE(t.empty());
  EXPECT_EQ(t.size(), 0);
}

TEST(EdgeCaseTest, ScalarTensor) {
  tt::tensor<double> s = {3.14};
  EXPECT_EQ(s.ndim(), 1);
  EXPECT_EQ(s.size(), 1);
  EXPECT_DOUBLE_EQ(s(0), 3.14);
}

TEST(EdgeCaseTest, NegativeIndices) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};

  auto v1 = tt::view(t, tt::range(-2, tt::_));
  EXPECT_EQ(v1.shape()[0], 2);
  EXPECT_EQ(v1(0), 4);
  EXPECT_EQ(v1(1), 5);

  auto v2 = tt::view(t, tt::range(0, -2));
  EXPECT_EQ(v2.shape()[0], 3);
  EXPECT_EQ(v2(0), 1);
}

TEST(EdgeCaseTest, ReverseSlicing) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};

  auto rev = tt::view(t, tt::range(4, tt::_, -1));
  EXPECT_EQ(rev.shape()[0], 5);
  EXPECT_EQ(rev(0), 5);
  EXPECT_EQ(rev(4), 1);
}

TEST(EdgeCaseTest, TensorFromView) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};
  auto v = tt::view(t, tt::all, tt::range(1, 3));

  tt::tensor<int> copy(v);
  EXPECT_EQ(copy.ndim(), 2);
  EXPECT_EQ(copy(0, 0), 2);
  EXPECT_EQ(copy(1, 1), 6);

  copy(0, 0) = 999;
  EXPECT_EQ(t(0, 1), 2);  // Original unchanged
}

TEST(EdgeCaseTest, BroadcastScalar) {
  tt::tensor<int> scalar = {42};
  auto bc = scalar.broadcast_to(tt::shape_t{3, 4});

  EXPECT_EQ(bc.ndim(), 2);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_EQ(bc(i, j), 42);
    }
  }
}

// =============================================================================
// Safety and Robustness Tests (Issue Fixes)
// =============================================================================

// Test 1: Negative-stride views work correctly with signed offsets
TEST(SafetyTest, NegativeStrideView) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};

  // Reverse the tensor using negative stride (tt::_ means go all the way to 0)
  auto rev = tt::view(t, tt::range(4, tt::_, -1));
  EXPECT_EQ(rev.shape()[0], 5);
  EXPECT_EQ(rev(0), 5);
  EXPECT_EQ(rev(1), 4);
  EXPECT_EQ(rev(2), 3);
  EXPECT_EQ(rev(3), 2);
  EXPECT_EQ(rev(4), 1);

  // Test iteration over reversed view
  std::vector<int> values;
  for (const auto& val : rev) {
    values.push_back(val);
  }
  EXPECT_EQ(values, (std::vector<int>{5, 4, 3, 2, 1}));
}

TEST(SafetyTest, NegativeStrideView2D) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  // Reverse rows
  auto rev_rows = tt::view(t, tt::range(2, tt::_, -1), tt::all);
  EXPECT_EQ(rev_rows(0, 0), 7);
  EXPECT_EQ(rev_rows(2, 0), 1);

  // Reverse columns
  auto rev_cols = tt::view(t, tt::all, tt::range(2, tt::_, -1));
  EXPECT_EQ(rev_cols(0, 0), 3);
  EXPECT_EQ(rev_cols(0, 2), 1);

  // Reverse both
  auto rev_both = tt::view(t, tt::range(2, tt::_, -1), tt::range(2, tt::_, -1));
  EXPECT_EQ(rev_both(0, 0), 9);
  EXPECT_EQ(rev_both(2, 2), 1);
}

TEST(SafetyTest, ChainedNegativeStrideViews) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};

  auto v1 = tt::view(t, tt::range(4, tt::_, -1));   // {5, 4, 3, 2, 1}
  auto v2 = tt::view(v1, tt::range(4, tt::_, -1));  // {1, 2, 3, 4, 5}

  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(v2(i), static_cast<int>(i + 1));
  }
}

// Test 2: broadcast_to throws on invalid rank
TEST(SafetyTest, BroadcastToLowerRankThrows) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};  // 2D tensor

  // Attempting to broadcast to lower rank should throw
  EXPECT_THROW(t.broadcast_to(tt::shape_t{6}), tt::broadcast_error);
}

TEST(SafetyTest, BroadcastToIncompatibleShapeThrows) {
  tt::tensor<int> t = {1, 2, 3};  // shape {3}

  // Cannot broadcast {3} to {2, 4} - dimensions don't match
  EXPECT_THROW(t.broadcast_to(tt::shape_t{2, 4}), tt::broadcast_error);
}

TEST(SafetyTest, BroadcastToValidHigherRank) {
  tt::tensor<int> t = {1, 2, 3};  // shape {3}

  // Can broadcast {3} to {2, 3}
  auto bc = t.broadcast_to(tt::shape_t{2, 3});
  EXPECT_EQ(bc.shape()[0], 2);
  EXPECT_EQ(bc.shape()[1], 3);
  EXPECT_EQ(bc(0, 0), 1);
  EXPECT_EQ(bc(1, 2), 3);
}

// Test 3: Per-dimension bounds checking
#ifndef NDEBUG
TEST(SafetyTest, IndexBoundsCheckTensor) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};  // 2x3

  // Valid indices
  EXPECT_NO_THROW(t(0, 0));
  EXPECT_NO_THROW(t(1, 2));

  // Out of bounds on first dimension
  EXPECT_THROW(t(2, 0), tt::tensor_error);

  // Out of bounds on second dimension
  EXPECT_THROW(t(0, 3), tt::tensor_error);

  // Both out of bounds
  EXPECT_THROW(t(5, 5), tt::tensor_error);
}

TEST(SafetyTest, IndexBoundsCheckView) {
  tt::tensor<int> t = {{1, 2, 3}, {4, 5, 6}};
  auto v = tt::view(t, tt::all, tt::range(0, 2));  // 2x2 view

  EXPECT_NO_THROW(v(0, 0));
  EXPECT_NO_THROW(v(1, 1));

  // Out of bounds for the view
  EXPECT_THROW(v(0, 2), tt::tensor_error);
  EXPECT_THROW(v(2, 0), tt::tensor_error);
}

TEST(SafetyTest, IndexBoundsCheckAt) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};

  std::vector<std::size_t> valid = {2};
  EXPECT_NO_THROW(t.at(valid));

  std::vector<std::size_t> invalid = {5};
  EXPECT_THROW(t.at(invalid), tt::tensor_error);
}
#endif

// Test 4: apply_slices ellipsis handling
TEST(SafetyTest, SingleEllipsis) {
  tt::tensor<int> t(tt::shape_t{2, 3, 4, 5});

  // Ellipsis should expand to fill remaining dimensions
  auto v1 = tt::view(t, tt::ellipsis, 0);  // Select first element of last dim
  EXPECT_EQ(v1.ndim(), 3);
  EXPECT_EQ(v1.shape()[0], 2);
  EXPECT_EQ(v1.shape()[1], 3);
  EXPECT_EQ(v1.shape()[2], 4);

  auto v2 = tt::view(t, 0, tt::ellipsis);  // Select first element of first dim
  EXPECT_EQ(v2.ndim(), 3);
  EXPECT_EQ(v2.shape()[0], 3);
  EXPECT_EQ(v2.shape()[1], 4);
  EXPECT_EQ(v2.shape()[2], 5);
}

#ifndef NDEBUG
TEST(SafetyTest, MultipleEllipsisThrows) {
  tt::tensor<int> t({2, 3, 4});

  // Multiple ellipsis should throw
  EXPECT_THROW(tt::view(t, tt::ellipsis, tt::ellipsis), tt::tensor_error);
}

TEST(SafetyTest, TooManySlicesThrows) {
  tt::tensor<int> t = {{1, 2}, {3, 4}};  // 2D

  // Too many indices for a 2D tensor (without ellipsis)
  EXPECT_THROW(tt::view(t, 0, 0, 0), tt::tensor_error);
}

TEST(SafetyTest, ZeroStepThrows) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};

  // Step == 0 should throw
  EXPECT_THROW(tt::view(t, tt::range(0, 5, 0)), tt::tensor_error);
}
#endif

// Test 5: Use uint8_t for boolean-like data (tensor<bool> is not supported)
TEST(SafetyTest, Uint8TensorAsFlags) {
  // tensor<bool> is explicitly disabled due to std::vector<bool> issues
  // Use uint8_t instead for boolean-like data
  tt::tensor<uint8_t> flags = {1, 0, 1, 0};

  EXPECT_EQ(flags.size(), 4);
  EXPECT_EQ(flags(0), 1);
  EXPECT_EQ(flags(1), 0);
  EXPECT_EQ(flags(2), 1);
  EXPECT_EQ(flags(3), 0);

  // Pointer access works correctly
  uint8_t* ptr = flags.data();
  EXPECT_EQ(ptr[0], 1);
  EXPECT_EQ(ptr[1], 0);

  // Modify through pointer
  ptr[1] = 1;
  EXPECT_EQ(flags(1), 1);
}

TEST(SafetyTest, Uint8TensorView) {
  tt::tensor<uint8_t> flags = {{1, 0}, {0, 1}};

  auto v = tt::view(flags, 0, tt::all);
  EXPECT_EQ(v.size(), 2);
  EXPECT_EQ(v(0), 1);
  EXPECT_EQ(v(1), 0);

  // Modify through view
  v(1) = 1;
  EXPECT_EQ(flags(0, 1), 1);
}

// Test signed offset handling for views
TEST(SafetyTest, ViewOffsetTypes) {
  tt::tensor<int> t = {1, 2, 3, 4, 5};

  // Create a view at an offset
  auto v = tt::view(t, tt::range(2, 5));
  EXPECT_EQ(v.size(), 3);
  EXPECT_EQ(v(0), 3);

  // The offset should be representable as signed
  auto offset = v.offset();
  EXPECT_GE(offset, 0);
}

// Test newaxis with ellipsis
TEST(SafetyTest, NewaxisWithEllipsis) {
  tt::tensor<int> t = {1, 2, 3};  // shape {3}

  // Add new axis at the beginning
  auto v1 = tt::view(t, tt::newaxis, tt::ellipsis);
  EXPECT_EQ(v1.ndim(), 2);
  EXPECT_EQ(v1.shape()[0], 1);
  EXPECT_EQ(v1.shape()[1], 3);

  // Add new axis at the end
  auto v2 = tt::view(t, tt::ellipsis, tt::newaxis);
  EXPECT_EQ(v2.ndim(), 2);
  EXPECT_EQ(v2.shape()[0], 3);
  EXPECT_EQ(v2.shape()[1], 1);
}

// =============================================================================
// Negative-Step Slice Tests (regression coverage)
// =============================================================================

TEST(SliceTest, NegativeStepWithLargerStep) {
  // range(4, tt::_, -2) with dim=5: indices 4, 2, 0 → 3 elements
  auto nr1 = tt::detail::normalize_range(tt::range(4, tt::_, -2), 5);
  EXPECT_EQ(nr1.start, 4);
  EXPECT_EQ(nr1.size, 3);
  EXPECT_EQ(nr1.step, -2);

  // range(tt::_, tt::_, -3) with dim=10: indices 9, 6, 3, 0 → 4 elements
  auto nr2 = tt::detail::normalize_range(tt::range(tt::_, tt::_, -3), 10);
  EXPECT_EQ(nr2.start, 9);
  EXPECT_EQ(nr2.size, 4);

  // range(9, 2, -3) with dim=10: indices 9, 6, 3 → 3 elements (stop=2 exclusive)
  auto nr3 = tt::detail::normalize_range(tt::range(9, 2, -3), 10);
  EXPECT_EQ(nr3.start, 9);
  EXPECT_EQ(nr3.size, 3);
}

TEST(SliceTest, NegativeStopNormalization) {
  // range(8, -5, -1) with dim=10: stop=-5 normalizes to 5, indices 8,7,6 → 3
  auto nr1 = tt::detail::normalize_range(tt::range(8, -5, -1), 10);
  EXPECT_EQ(nr1.start, 8);
  EXPECT_EQ(nr1.size, 3);

  // range(9, -3, -1) with dim=10: stop=-3 → 7, indices 9,8 → 2
  auto nr2 = tt::detail::normalize_range(tt::range(9, -3, -1), 10);
  EXPECT_EQ(nr2.start, 9);
  EXPECT_EQ(nr2.size, 2);

  // range(4, -1, -2) with dim=5: stop=-1 → 4, start==stop → empty
  auto nr3 = tt::detail::normalize_range(tt::range(4, -1, -2), 5);
  EXPECT_EQ(nr3.size, 0);
}

TEST(SliceTest, NegativeStepViewValues) {
  tt::tensor<int> t = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // Step -2 from end: indices 9, 7, 5, 3, 1
  auto v1 = tt::view(t, tt::range(tt::_, tt::_, -2));
  EXPECT_EQ(v1.shape()[0], 5);
  EXPECT_EQ(v1(0), 9);
  EXPECT_EQ(v1(1), 7);
  EXPECT_EQ(v1(2), 5);
  EXPECT_EQ(v1(3), 3);
  EXPECT_EQ(v1(4), 1);

  // range(8, -5, -1) on dim=10: stop normalizes to 5 → indices 8, 7, 6
  auto v2 = tt::view(t, tt::range(8, -5, -1));
  EXPECT_EQ(v2.shape()[0], 3);
  EXPECT_EQ(v2(0), 8);
  EXPECT_EQ(v2(1), 7);
  EXPECT_EQ(v2(2), 6);

  // range(4, -1, -2) on dim=5 subset: stop=-1 → 4, start==stop → empty
  tt::tensor<int> t2 = {0, 1, 2, 3, 4};
  auto v3 = tt::view(t2, tt::range(4, -1, -2));
  EXPECT_EQ(v3.shape()[0], 0);
}

// =============================================================================
// Column-Major Correctness Tests
// =============================================================================

TEST(ColumnMajorTest, UnaryOp) {
  tt::tensor<int> t(tt::shape_t{2, 3}, tt::layout_type::column_major);
  // Fill with logical values: t(i,j) = i*3 + j
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      t(i, j) = static_cast<int>(i * 3 + j);
    }
  }

  auto neg = -t;
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(neg(i, j), -static_cast<int>(i * 3 + j));
    }
  }
}

TEST(ColumnMajorTest, ScalarOp) {
  tt::tensor<int> t(tt::shape_t{2, 3}, tt::layout_type::column_major);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      t(i, j) = static_cast<int>(i * 3 + j + 1);
    }
  }

  auto result = t * 10;
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(result(i, j), static_cast<int>((i * 3 + j + 1) * 10));
    }
  }

  auto result2 = 100 - t;
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(result2(i, j), 100 - static_cast<int>(i * 3 + j + 1));
    }
  }
}

TEST(ColumnMajorTest, CopyFromMismatchedLayout) {
  tt::tensor<int> col(tt::shape_t{2, 3}, tt::layout_type::column_major);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      col(i, j) = static_cast<int>(i * 3 + j);
    }
  }

  tt::tensor<int> row(tt::shape_t{2, 3}, tt::layout_type::row_major);
  row.copy_from(col);

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(row(i, j), static_cast<int>(i * 3 + j));
    }
  }
}

TEST(ColumnMajorTest, ViewIsContiguous) {
  tt::tensor<int> t(tt::shape_t{3, 4}, tt::layout_type::column_major);
  auto v = t.view();
  EXPECT_TRUE(v.is_contiguous());
}

TEST(ColumnMajorTest, PartialViewNotContiguous) {
  tt::tensor<int> t(tt::shape_t{3, 4}, tt::layout_type::column_major);
  // Selecting a subset of rows from a column-major tensor is not contiguous
  auto v = tt::view(t, tt::range(0, 2), tt::all);
  EXPECT_FALSE(v.is_contiguous());
}

}  // namespace
