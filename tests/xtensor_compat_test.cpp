// Tests adapted from xtensor test suite
// Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht
// Copyright (c) QuantStack
// Distributed under the terms of the BSD 3-Clause License.

// Define _USE_MATH_DEFINES before including cmath for M_PI on MSVC
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <numeric>

#include "../include/tinytensor/tinytensor.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// =============================================================================
// Builder Tests (adapted from test_xbuilder.cpp)
// =============================================================================

TEST(XtensorCompat_Builder, Ones) {
  auto m = tt::ones<double>(tt::shape_t{1, 2});
  EXPECT_EQ(m.ndim(), 2u);
  EXPECT_DOUBLE_EQ(m(0, 1), 1.0);
}

TEST(XtensorCompat_Builder, ArangeSimple) {
  auto ls = tt::arange<double>(0, 50);
  EXPECT_EQ(ls.ndim(), 1u);
  EXPECT_EQ(ls.shape()[0], 50u);
  EXPECT_DOUBLE_EQ(ls(0), 0);
  EXPECT_DOUBLE_EQ(ls(49), 49);
  EXPECT_DOUBLE_EQ(ls(29), 29);
}

TEST(XtensorCompat_Builder, ArangeReshape) {
  auto ls = tt::arange<double>(0, 50);
  ls.reshape(tt::shape_t{5, 10});

  EXPECT_EQ(ls.ndim(), 2u);
  EXPECT_EQ(ls.shape()[0], 5u);
  EXPECT_EQ(ls.shape()[1], 10u);
  EXPECT_DOUBLE_EQ(ls(4, 9), 49);
}

TEST(XtensorCompat_Builder, ArangeMinMax) {
  auto ls = tt::arange<unsigned int>(10u, 20u);
  EXPECT_EQ(ls.ndim(), 1u);
  EXPECT_EQ(ls.shape()[0], 10u);
  EXPECT_EQ(ls(0), 10u);
  EXPECT_EQ(ls(9), 19u);
  EXPECT_EQ(ls(2), 12u);
}

TEST(XtensorCompat_Builder, ArangeMinMaxStep) {
  auto ls = tt::arange<float>(10.0f, 20.0f, 0.5f);
  EXPECT_EQ(ls.ndim(), 1u);
  EXPECT_EQ(ls.shape()[0], 20u);
  EXPECT_FLOAT_EQ(ls(0), 10.f);
  EXPECT_FLOAT_EQ(ls(10), 15.f);
  EXPECT_FLOAT_EQ(ls(3), 11.5f);

  auto l4 = tt::arange<int>(0, 10, 3);
  EXPECT_EQ(l4.shape()[0], 4u);
  EXPECT_EQ(l4(0), 0);
  EXPECT_EQ(l4(1), 3);
  EXPECT_EQ(l4(2), 6);
  EXPECT_EQ(l4(3), 9);
}

TEST(XtensorCompat_Builder, ArangeReverse) {
  auto a1 = tt::arange(8, 5, -1);
  EXPECT_EQ(a1.ndim(), 1u);
  EXPECT_EQ(a1.shape()[0], 3u);
  EXPECT_EQ(a1(0), 8);
  EXPECT_EQ(a1(1), 7);
  EXPECT_EQ(a1(2), 6);
}

TEST(XtensorCompat_Builder, Linspace) {
  auto ls = tt::linspace<float>(20.f, 50.f, 50);
  EXPECT_EQ(ls.ndim(), 1u);
  EXPECT_EQ(ls.shape()[0], 50u);
  EXPECT_FLOAT_EQ(ls(0), 20.f);
  EXPECT_FLOAT_EQ(ls(49), 50.f);

  float at_3 = 20 + 3 * (50.f - 20.f) / (50.f - 1.f);
  EXPECT_FLOAT_EQ(ls(3), at_3);
}

TEST(XtensorCompat_Builder, LinspaceReshape) {
  auto a = tt::linspace<double>(20., 50., 50);
  a.reshape(tt::shape_t{5, 10});
  EXPECT_EQ(a.ndim(), 2u);
  EXPECT_EQ(a.shape()[0], 5u);
  EXPECT_EQ(a.shape()[1], 10u);
  EXPECT_DOUBLE_EQ(a(0, 0), 20.);
  EXPECT_DOUBLE_EQ(a(4, 9), 50.);
}

TEST(XtensorCompat_Builder, Eye) {
  auto e = tt::eye<double>(5);
  EXPECT_EQ(e.ndim(), 2u);
  EXPECT_EQ(e.shape()[0], 5u);
  EXPECT_EQ(e.shape()[1], 5u);

  EXPECT_DOUBLE_EQ(e(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(e(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(e(2, 2), 1.0);
  EXPECT_DOUBLE_EQ(e(4, 2), 0.0);
}

TEST(XtensorCompat_Builder, Diag) {
  tt::tensor<double> v = {1, 5, 9};
  auto d = tt::diag(v);

  EXPECT_EQ(d.ndim(), 2u);
  EXPECT_EQ(d.shape()[0], 3u);
  EXPECT_EQ(d.shape()[1], 3u);
  EXPECT_DOUBLE_EQ(d(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(d(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(d(2, 2), 9.0);
  EXPECT_DOUBLE_EQ(d(0, 1), 0.0);
}

// =============================================================================
// Broadcast Tests (adapted from test_xbroadcast.cpp)
// =============================================================================

TEST(XtensorCompat_Broadcast, Basic) {
  tt::tensor<double> m1 = {{1, 2, 3}, {4, 5, 6}};

  auto m1_broadcast = m1.broadcast_to(tt::shape_t{1, 2, 3});
  EXPECT_DOUBLE_EQ(m1_broadcast(0, 0, 0), 1.0);
  EXPECT_DOUBLE_EQ(m1_broadcast(0, 1, 0), 4.0);
  EXPECT_DOUBLE_EQ(m1_broadcast(0, 1, 1), 5.0);
}

TEST(XtensorCompat_Broadcast, Element) {
  tt::tensor<double> m1 = {{1, 2, 3}, {4, 5, 6}};

  auto m1_broadcast = m1.broadcast_to(tt::shape_t{4, 2, 3});
  EXPECT_DOUBLE_EQ(m1_broadcast(0, 1, 1), 5.0);
  EXPECT_DOUBLE_EQ(m1_broadcast(3, 1, 1), 5.0);  // broadcast along first dim
}

TEST(XtensorCompat_Broadcast, Iterator) {
  tt::tensor<int> m1 = {1, 2, 3};
  auto m1_broadcast = m1.broadcast_to(tt::shape_t{2, 3});

  auto iter = m1_broadcast.begin();
  EXPECT_EQ(*iter, 1);
  ++iter;
  EXPECT_EQ(*iter, 2);
  ++iter;
  EXPECT_EQ(*iter, 3);
  ++iter;
  EXPECT_EQ(*iter, 1);  // wrapped
}

// =============================================================================
// Strided View Tests (adapted from test_xstrided_view.cpp)
// =============================================================================

TEST(XtensorCompat_StridedView, Simple) {
  tt::tensor<double> a(tt::shape_t{3, 4});
  std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::copy(data.begin(), data.end(), a.data());

  auto view1 = tt::view(a, std::ptrdiff_t{1}, tt::range(1, 4));
  EXPECT_DOUBLE_EQ(a(1, 1), view1(0));
  EXPECT_DOUBLE_EQ(a(1, 2), view1(1));
  EXPECT_EQ(view1.ndim(), 1u);

  auto view0 = tt::view(a, std::ptrdiff_t{0}, tt::range(0, 3));
  EXPECT_DOUBLE_EQ(a(0, 0), view0(0));
  EXPECT_DOUBLE_EQ(a(0, 1), view0(1));
  EXPECT_EQ(view0.ndim(), 1u);
  EXPECT_EQ(view0.shape()[0], 3u);

  auto view2 = tt::view(a, tt::range(0, 2), std::ptrdiff_t{2});
  EXPECT_DOUBLE_EQ(a(0, 2), view2(0));
  EXPECT_DOUBLE_EQ(a(1, 2), view2(1));
  EXPECT_EQ(view2.ndim(), 1u);
  EXPECT_EQ(view2.shape()[0], 2u);

  auto view6 = tt::view(a, std::ptrdiff_t{1}, tt::all);
  EXPECT_DOUBLE_EQ(a(1, 0), view6(0));
  EXPECT_DOUBLE_EQ(a(1, 1), view6(1));
  EXPECT_DOUBLE_EQ(a(1, 2), view6(2));
  EXPECT_DOUBLE_EQ(a(1, 3), view6(3));

  auto view7 = tt::view(a, tt::all, std::ptrdiff_t{2});
  EXPECT_DOUBLE_EQ(a(0, 2), view7(0));
  EXPECT_DOUBLE_EQ(a(1, 2), view7(1));
  EXPECT_DOUBLE_EQ(a(2, 2), view7(2));
}

TEST(XtensorCompat_StridedView, ThreeDimensional) {
  tt::tensor<double> a(tt::shape_t{3, 4, 2});
  std::vector<double> data = {1,  2,  3,  4,  5,  6,   7,   8,
                              9,  10, 11, 12, 21, 22,  23,  24,
                              25, 26, 27, 28, 29, 210, 211, 212};
  std::copy(data.begin(), data.end(), a.data());

  auto view1 = tt::view(a, std::ptrdiff_t{1}, tt::all, tt::all);
  EXPECT_EQ(view1.ndim(), 2u);
  EXPECT_EQ(view1.shape()[0], 4u);
  EXPECT_EQ(view1.shape()[1], 2u);
  EXPECT_DOUBLE_EQ(a(1, 0, 0), view1(0, 0));
  EXPECT_DOUBLE_EQ(a(1, 0, 1), view1(0, 1));
  EXPECT_DOUBLE_EQ(a(1, 1, 0), view1(1, 0));
  EXPECT_DOUBLE_EQ(a(1, 1, 1), view1(1, 1));
}

TEST(XtensorCompat_StridedView, Iterator) {
  tt::tensor<double> a(tt::shape_t{2, 3, 4});
  std::vector<double> data{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  std::copy(data.begin(), data.end(), a.data());

  auto view1 = tt::view(a, tt::range(0, 2), std::ptrdiff_t{1}, tt::range(1, 4));
  auto iter = view1.begin();

  EXPECT_DOUBLE_EQ(6, *iter);
  ++iter;
  EXPECT_DOUBLE_EQ(7, *iter);
  ++iter;
  EXPECT_DOUBLE_EQ(8, *iter);
  ++iter;
  EXPECT_DOUBLE_EQ(18, *iter);
  ++iter;
  EXPECT_DOUBLE_EQ(19, *iter);
  ++iter;
  EXPECT_DOUBLE_EQ(20, *iter);
}

TEST(XtensorCompat_StridedView, Newaxis) {
  tt::tensor<double> a(tt::shape_t{3, 4});
  std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::copy(data.begin(), data.end(), a.data());

  auto view1 = tt::view(a, tt::all, tt::newaxis, tt::all);
  EXPECT_DOUBLE_EQ(a(1, 1), view1(1, 0, 1));
  EXPECT_DOUBLE_EQ(a(1, 2), view1(1, 0, 2));
  EXPECT_EQ(view1.ndim(), 3u);
  EXPECT_EQ(view1.shape()[0], 3u);
  EXPECT_EQ(view1.shape()[1], 1u);
  EXPECT_EQ(view1.shape()[2], 4u);

  auto view2 = tt::view(a, tt::all, tt::all, tt::newaxis);
  EXPECT_DOUBLE_EQ(a(1, 1), view2(1, 1, 0));
  EXPECT_DOUBLE_EQ(a(1, 2), view2(1, 2, 0));
  EXPECT_EQ(view2.ndim(), 3u);
  EXPECT_EQ(view2.shape()[0], 3u);
  EXPECT_EQ(view2.shape()[1], 4u);
  EXPECT_EQ(view2.shape()[2], 1u);

  auto view3 = tt::view(a, std::ptrdiff_t{1}, tt::newaxis, tt::all);
  EXPECT_DOUBLE_EQ(a(1, 1), view3(0, 1));
  EXPECT_DOUBLE_EQ(a(1, 2), view3(0, 2));
  EXPECT_EQ(view3.ndim(), 2u);
}

TEST(XtensorCompat_StridedView, RangeAdaptor) {
  tt::tensor<int> a = {1, 2, 3, 4, 5};

  auto v1 = tt::view(a, tt::range(3, tt::_));
  tt::tensor<int> v1e = {4, 5};
  EXPECT_EQ(v1.shape()[0], 2u);
  EXPECT_EQ(v1(0), 4);
  EXPECT_EQ(v1(1), 5);

  auto v2 = tt::view(a, tt::range(0, 2));
  EXPECT_EQ(v2.shape()[0], 2u);
  EXPECT_EQ(v2(0), 1);
  EXPECT_EQ(v2(1), 2);

  auto v4 = tt::view(a, tt::range(4, tt::_, -1));
  EXPECT_EQ(v4.shape()[0], 5u);
  EXPECT_EQ(v4(0), 5);
  EXPECT_EQ(v4(1), 4);

  auto v5 = tt::view(a, tt::range(2, tt::_, -1));
  EXPECT_EQ(v5.shape()[0], 3u);
  EXPECT_EQ(v5(0), 3);
  EXPECT_EQ(v5(1), 2);
  EXPECT_EQ(v5(2), 1);

  auto v7 = tt::view(a, tt::range(1, tt::_, 2));
  EXPECT_EQ(v7.shape()[0], 2u);
  EXPECT_EQ(v7(0), 2);
  EXPECT_EQ(v7(1), 4);

  auto v8 = tt::view(a, tt::range(2, tt::_, 2));
  EXPECT_EQ(v8.shape()[0], 2u);
  EXPECT_EQ(v8(0), 3);
  EXPECT_EQ(v8(1), 5);
}

TEST(XtensorCompat_StridedView, Ellipsis) {
  tt::tensor<int> a(tt::shape_t{5, 5, 1, 1, 1, 4});
  std::iota(a.begin(), a.end(), 0);

  auto v1 = tt::view(a, std::ptrdiff_t{1}, std::ptrdiff_t{1}, tt::ellipsis);
  EXPECT_EQ(v1.ndim(), 4u);
  EXPECT_EQ(v1.shape()[0], 1u);
  EXPECT_EQ(v1.shape()[1], 1u);
  EXPECT_EQ(v1.shape()[2], 1u);
  EXPECT_EQ(v1.shape()[3], 4u);
}

TEST(XtensorCompat_StridedView, ViewOnView) {
  tt::tensor<int> a = tt::ones<int>(tt::shape_t{3, 4, 5});
  auto v1 = tt::view(a, std::ptrdiff_t{1}, tt::all, tt::all);
  auto vv1 = tt::view(v1, std::ptrdiff_t{1}, tt::all);

  // Set values through chained view
  for (std::size_t i = 0; i < 5; ++i) {
    vv1(i) = 5;
  }

  EXPECT_EQ(a(0, 0, 0), 1);
  EXPECT_EQ(a(1, 1, 0), 5);
  EXPECT_EQ(a(1, 1, 4), 5);
  EXPECT_EQ(a(1, 2, 4), 1);
  EXPECT_EQ(v1(1, 4), 5);
}

// =============================================================================
// Manipulation Tests (adapted from test_xmanipulation.cpp)
// =============================================================================

TEST(XtensorCompat_Manipulation, TransposeBasic) {
  tt::tensor<int> a = {{0, 1, 2}, {3, 4, 5}};
  auto tr = a.transposed();

  EXPECT_EQ(tr.shape()[0], 3u);
  EXPECT_EQ(tr.shape()[1], 2u);
  EXPECT_EQ(a(0, 0), tr(0, 0));
  EXPECT_EQ(a(0, 1), tr(1, 0));
  EXPECT_EQ(a(0, 2), tr(2, 0));
  EXPECT_EQ(a(1, 0), tr(0, 1));
  EXPECT_EQ(a(1, 1), tr(1, 1));
  EXPECT_EQ(a(1, 2), tr(2, 1));
}

TEST(XtensorCompat_Manipulation, Flatten) {
  tt::tensor<int> a = {{0, 1, 2}, {3, 4, 5}};

  auto flat = a.flattened();
  EXPECT_EQ(flat(0), a(0, 0));
  EXPECT_EQ(flat(1), a(0, 1));
  EXPECT_EQ(flat(2), a(0, 2));
  EXPECT_EQ(flat(3), a(1, 0));
  EXPECT_EQ(flat(4), a(1, 1));
  EXPECT_EQ(flat(5), a(1, 2));
}

TEST(XtensorCompat_Manipulation, Squeeze) {
  tt::tensor<double> b(tt::shape_t{3, 3, 1, 1, 2, 1, 3});
  std::iota(b.begin(), b.end(), 0);

  auto sq = b.copy();
  sq.squeeze();

  EXPECT_EQ(sq.ndim(), 4u);
  EXPECT_EQ(sq.shape()[0], 3u);
  EXPECT_EQ(sq.shape()[1], 3u);
  EXPECT_EQ(sq.shape()[2], 2u);
  EXPECT_EQ(sq.shape()[3], 3u);
}

TEST(XtensorCompat_Manipulation, Unsqueeze) {
  tt::tensor<int> a = {1, 2, 3};

  a.unsqueeze(0);
  EXPECT_EQ(a.ndim(), 2u);
  EXPECT_EQ(a.shape()[0], 1u);
  EXPECT_EQ(a.shape()[1], 3u);

  a.unsqueeze(2);
  EXPECT_EQ(a.ndim(), 3u);
  EXPECT_EQ(a.shape()[2], 1u);
}

TEST(XtensorCompat_Manipulation, ReshapeInference) {
  tt::tensor<int> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  a.reshape({3, -1});
  EXPECT_EQ(a.ndim(), 2u);
  EXPECT_EQ(a.shape()[0], 3u);
  EXPECT_EQ(a.shape()[1], 4u);

  a.reshape({-1, 2, 2});
  EXPECT_EQ(a.ndim(), 3u);
  EXPECT_EQ(a.shape()[0], 3u);
  EXPECT_EQ(a.shape()[1], 2u);
  EXPECT_EQ(a.shape()[2], 2u);
}

// =============================================================================
// Math Tests (adapted from test_xmath.cpp)
// =============================================================================

TEST(XtensorCompat_Math, Abs) {
  tt::tensor<double> a(tt::shape_t{3, 2}, -4.5);
  auto result = tt::abs(a);
  EXPECT_DOUBLE_EQ(result(0, 0), std::abs(a(0, 0)));
  EXPECT_DOUBLE_EQ(result(0, 0), 4.5);
}

TEST(XtensorCompat_Math, Sqrt) {
  tt::tensor<double> a = {1.0, 4.0, 9.0, 16.0, 25.0};
  auto result = tt::sqrt(a);
  EXPECT_DOUBLE_EQ(result(0), 1.0);
  EXPECT_DOUBLE_EQ(result(1), 2.0);
  EXPECT_DOUBLE_EQ(result(2), 3.0);
  EXPECT_DOUBLE_EQ(result(3), 4.0);
  EXPECT_DOUBLE_EQ(result(4), 5.0);
}

TEST(XtensorCompat_Math, ExpLog) {
  tt::tensor<double> a = {0.0, 1.0, 2.0};
  auto exp_result = tt::exp(a);
  EXPECT_NEAR(exp_result(0), 1.0, 1e-10);
  EXPECT_NEAR(exp_result(1), std::exp(1.0), 1e-10);
  EXPECT_NEAR(exp_result(2), std::exp(2.0), 1e-10);

  auto log_result = tt::log(exp_result);
  EXPECT_NEAR(log_result(0), 0.0, 1e-10);
  EXPECT_NEAR(log_result(1), 1.0, 1e-10);
  EXPECT_NEAR(log_result(2), 2.0, 1e-10);
}

TEST(XtensorCompat_Math, Trigonometric) {
  tt::tensor<double> a = {0.0, M_PI / 6, M_PI / 4, M_PI / 3, M_PI / 2};

  auto sin_result = tt::sin(a);
  EXPECT_NEAR(sin_result(0), 0.0, 1e-10);
  EXPECT_NEAR(sin_result(1), 0.5, 1e-10);
  EXPECT_NEAR(sin_result(4), 1.0, 1e-10);

  auto cos_result = tt::cos(a);
  EXPECT_NEAR(cos_result(0), 1.0, 1e-10);
  EXPECT_NEAR(cos_result(1), std::sqrt(3.0) / 2.0, 1e-10);
  EXPECT_NEAR(cos_result(4), 0.0, 1e-10);
}

TEST(XtensorCompat_Math, Pow) {
  tt::tensor<double> base = {2.0, 3.0, 4.0};
  tt::tensor<double> exp = {2.0, 2.0, 2.0};

  auto result = tt::pow(base, exp);
  EXPECT_DOUBLE_EQ(result(0), 4.0);
  EXPECT_DOUBLE_EQ(result(1), 9.0);
  EXPECT_DOUBLE_EQ(result(2), 16.0);

  auto result2 = tt::pow(base, 3.0);
  EXPECT_DOUBLE_EQ(result2(0), 8.0);
  EXPECT_DOUBLE_EQ(result2(1), 27.0);
  EXPECT_DOUBLE_EQ(result2(2), 64.0);
}

TEST(XtensorCompat_Math, MinMax) {
  tt::tensor<double> a = {-10.0};
  EXPECT_DOUBLE_EQ(tt::min(a), -10.0);
  EXPECT_DOUBLE_EQ(tt::max(a), -10.0);

  tt::tensor<double> b = {-10.0, -20.0};
  EXPECT_DOUBLE_EQ(tt::min(b), -20.0);
  EXPECT_DOUBLE_EQ(tt::max(b), -10.0);

  tt::tensor<double> c = {-10.0, +20.0};
  EXPECT_DOUBLE_EQ(tt::min(c), -10.0);
  EXPECT_DOUBLE_EQ(tt::max(c), +20.0);
}

TEST(XtensorCompat_Math, SumProd) {
  tt::tensor<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
  EXPECT_DOUBLE_EQ(tt::sum(a), 15.0);
  EXPECT_DOUBLE_EQ(tt::prod(a), 120.0);
  EXPECT_DOUBLE_EQ(tt::mean(a), 3.0);
}

// =============================================================================
// Operation Tests
// =============================================================================

TEST(XtensorCompat_Operation, ArithmeticOperators) {
  tt::tensor<double> a = {{1, 2, 3}, {4, 5, 6}};
  tt::tensor<double> b = {{1, 2, 3}, {4, 5, 6}};

  auto sum = a + b;
  EXPECT_DOUBLE_EQ(sum(0, 0), 2);
  EXPECT_DOUBLE_EQ(sum(0, 1), 4);
  EXPECT_DOUBLE_EQ(sum(1, 2), 12);

  auto diff = a - b;
  EXPECT_DOUBLE_EQ(diff(0, 0), 0);
  EXPECT_DOUBLE_EQ(diff(1, 2), 0);

  auto prod = a * b;
  EXPECT_DOUBLE_EQ(prod(0, 0), 1);
  EXPECT_DOUBLE_EQ(prod(1, 2), 36);
}

TEST(XtensorCompat_Operation, BroadcastArithmetic) {
  tt::tensor<double> a = {{1, 2, 3}, {4, 5, 6}};
  tt::tensor<double> b = {10, 20, 30};

  auto result = a + b;
  EXPECT_EQ(result.shape()[0], 2u);
  EXPECT_EQ(result.shape()[1], 3u);
  EXPECT_DOUBLE_EQ(result(0, 0), 11.0);
  EXPECT_DOUBLE_EQ(result(0, 1), 22.0);
  EXPECT_DOUBLE_EQ(result(0, 2), 33.0);
  EXPECT_DOUBLE_EQ(result(1, 0), 14.0);
  EXPECT_DOUBLE_EQ(result(1, 1), 25.0);
  EXPECT_DOUBLE_EQ(result(1, 2), 36.0);
}

TEST(XtensorCompat_Operation, ScalarOperations) {
  tt::tensor<double> a = {1.0, 2.0, 3.0};

  auto add = a + 10.0;
  EXPECT_DOUBLE_EQ(add(0), 11.0);
  EXPECT_DOUBLE_EQ(add(1), 12.0);
  EXPECT_DOUBLE_EQ(add(2), 13.0);

  auto mul = a * 2.0;
  EXPECT_DOUBLE_EQ(mul(0), 2.0);
  EXPECT_DOUBLE_EQ(mul(1), 4.0);
  EXPECT_DOUBLE_EQ(mul(2), 6.0);

  auto sub = 10.0 - a;
  EXPECT_DOUBLE_EQ(sub(0), 9.0);
  EXPECT_DOUBLE_EQ(sub(1), 8.0);
  EXPECT_DOUBLE_EQ(sub(2), 7.0);
}

TEST(XtensorCompat_Operation, UnaryMinus) {
  tt::tensor<double> a = {1.0, -2.0, 3.0};
  auto neg = -a;
  EXPECT_DOUBLE_EQ(neg(0), -1.0);
  EXPECT_DOUBLE_EQ(neg(1), 2.0);
  EXPECT_DOUBLE_EQ(neg(2), -3.0);
}

// =============================================================================
// Fixed Tensor Tests (adapted from test_xfixed.cpp)
// =============================================================================

TEST(XtensorCompat_Fixed, Basic) {
  tt::fixed_tensor<double, 3, 4, 5> a(1.0);

  EXPECT_EQ(a.ndim(), 3u);
  EXPECT_EQ(a.size(), 60u);
  EXPECT_EQ(a.shape()[0], 3u);
  EXPECT_EQ(a.shape()[1], 4u);
  EXPECT_EQ(a.shape()[2], 5u);

  a(1, 2, 3) = 42.0;
  EXPECT_DOUBLE_EQ(a(1, 2, 3), 42.0);
  EXPECT_DOUBLE_EQ(a(0, 0, 0), 1.0);
}

TEST(XtensorCompat_Fixed, CompileTimeStrides) {
  using tensor_type = tt::fixed_tensor<double, 3, 4, 5>;

  static_assert(tensor_type::ndim_v == 3);
  static_assert(tensor_type::size_v == 60);
  static_assert(tensor_type::shape_v[0] == 3);
  static_assert(tensor_type::shape_v[1] == 4);
  static_assert(tensor_type::shape_v[2] == 5);
  // Row major: strides are 20, 5, 1
  static_assert(tensor_type::strides_v[0] == 20);
  static_assert(tensor_type::strides_v[1] == 5);
  static_assert(tensor_type::strides_v[2] == 1);
}

TEST(XtensorCompat_Fixed, Iteration) {
  tt::fixed_tensor<int, 2, 3> a;
  std::iota(a.begin(), a.end(), 0);

  EXPECT_EQ(a(0, 0), 0);
  EXPECT_EQ(a(0, 1), 1);
  EXPECT_EQ(a(0, 2), 2);
  EXPECT_EQ(a(1, 0), 3);
  EXPECT_EQ(a(1, 1), 4);
  EXPECT_EQ(a(1, 2), 5);
}

// =============================================================================
// Additional View Tests (adapted from test_xview.cpp)
// =============================================================================

TEST(XtensorCompat_View, NegativeIndex) {
  tt::tensor<double> a(tt::shape_t{3, 4});
  std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::copy(data.begin(), data.end(), a.data());

  // -2 should be same as index 1 for size 3
  auto view0 = tt::view(a, std::ptrdiff_t{-2}, tt::range(1, 4));
  auto view1 = tt::view(a, std::ptrdiff_t{1}, tt::range(1, 4));

  EXPECT_EQ(view0.shape()[0], view1.shape()[0]);
  EXPECT_DOUBLE_EQ(view0(0), view1(0));
  EXPECT_DOUBLE_EQ(view0(1), view1(1));
  EXPECT_DOUBLE_EQ(view0(2), view1(2));
}

TEST(XtensorCompat_View, ConstTensor) {
  const tt::tensor<int> a = {{0, 1}, {2, 3}};
  auto v = tt::view(a, std::ptrdiff_t{1}, tt::range(1, 2));

  int val = v(0);
  EXPECT_EQ(val, 3);

  auto it = v.begin();
  EXPECT_EQ(*it, val);
}

TEST(XtensorCompat_View, ThreeDimensionalComplex) {
  tt::tensor<double> a(tt::shape_t{3, 4, 2});
  std::vector<double> data = {1,  2,  3,  4,  5,  6,   7,   8,
                              9,  10, 11, 12, 21, 22,  23,  24,
                              25, 26, 27, 28, 29, 210, 211, 212};
  std::copy(data.begin(), data.end(), a.data());

  auto view1 = tt::view(a, std::ptrdiff_t{1});
  EXPECT_EQ(view1.ndim(), 2u);
  EXPECT_DOUBLE_EQ(a(1, 0, 0), view1(0, 0));
  EXPECT_DOUBLE_EQ(a(1, 0, 1), view1(0, 1));
  EXPECT_DOUBLE_EQ(a(1, 1, 0), view1(1, 0));
  EXPECT_DOUBLE_EQ(a(1, 1, 1), view1(1, 1));
}

TEST(XtensorCompat_View, NewaxisIterating) {
  tt::tensor<double> a(tt::shape_t{3, 4});
  std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::copy(data.begin(), data.end(), a.data());

  auto view1 = tt::view(a, tt::all, tt::all, tt::newaxis);
  auto iter1 = view1.begin();

  EXPECT_DOUBLE_EQ(a(0, 0), *iter1);
  ++iter1;
  EXPECT_DOUBLE_EQ(a(0, 1), *iter1);
  ++iter1;
  EXPECT_DOUBLE_EQ(a(0, 2), *iter1);
  ++iter1;
  EXPECT_DOUBLE_EQ(a(0, 3), *iter1);
  ++iter1;
  EXPECT_DOUBLE_EQ(a(1, 0), *iter1);
  ++iter1;
  EXPECT_DOUBLE_EQ(a(1, 1), *iter1);
}

TEST(XtensorCompat_View, NewaxisBroadcast) {
  tt::tensor<double> a(tt::shape_t{3, 4});
  std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::copy(data.begin(), data.end(), a.data());

  tt::tensor<double> b = {1, 2, 3, 4};

  auto v = tt::view(b, tt::newaxis, tt::all);
  // v is now shape {1, 4}
  EXPECT_EQ(v.ndim(), 2u);
  EXPECT_EQ(v.shape()[0], 1u);
  EXPECT_EQ(v.shape()[1], 4u);

  // Can be used in broadcast arithmetic
  // a + v should broadcast {3, 4} + {1, 4} -> {3, 4}
  // But we don't have view arithmetic yet, so just test the view shape
}

TEST(XtensorCompat_View, RangeWithNegativeStep) {
  tt::tensor<int> a = {1, 2, 3, 4, 5};

  // Reverse entire array (use tt::_ to go all the way to index 0)
  auto v1 = tt::view(a, tt::range(4, tt::_, -1));
  EXPECT_EQ(v1.shape()[0], 5u);
  EXPECT_EQ(v1(0), 5);
  EXPECT_EQ(v1(1), 4);
  EXPECT_EQ(v1(2), 3);
  EXPECT_EQ(v1(3), 2);
  EXPECT_EQ(v1(4), 1);

  // Partial reverse
  auto v2 = tt::view(a, tt::range(3, 0, -1));
  EXPECT_EQ(v2.shape()[0], 3u);
  EXPECT_EQ(v2(0), 4);
  EXPECT_EQ(v2(1), 3);
  EXPECT_EQ(v2(2), 2);
}

TEST(XtensorCompat_View, ViewElementAssignment) {
  tt::tensor<double> a(tt::shape_t{3, 4});
  std::iota(a.begin(), a.end(), 1);

  auto v = tt::view(a, std::ptrdiff_t{1}, tt::all);

  // Modify through view
  v(0) = 100;
  v(1) = 200;
  v(2) = 300;
  v(3) = 400;

  EXPECT_DOUBLE_EQ(a(1, 0), 100);
  EXPECT_DOUBLE_EQ(a(1, 1), 200);
  EXPECT_DOUBLE_EQ(a(1, 2), 300);
  EXPECT_DOUBLE_EQ(a(1, 3), 400);

  // Other rows unchanged
  EXPECT_DOUBLE_EQ(a(0, 0), 1);
  EXPECT_DOUBLE_EQ(a(2, 0), 9);
}

TEST(XtensorCompat_View, ViewChainedSlicing) {
  tt::tensor<int> a(tt::shape_t{4, 5, 6});
  std::iota(a.begin(), a.end(), 0);

  // First slice: select row 1, keep all columns/depths
  auto v1 = tt::view(a, std::ptrdiff_t{1}, tt::all, tt::all);
  EXPECT_EQ(v1.ndim(), 2u);
  EXPECT_EQ(v1.shape()[0], 5u);
  EXPECT_EQ(v1.shape()[1], 6u);

  // Second slice on the view
  auto v2 = tt::view(v1, std::ptrdiff_t{2}, tt::all);
  EXPECT_EQ(v2.ndim(), 1u);
  EXPECT_EQ(v2.shape()[0], 6u);

  // Values should match original
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(v2(i), a(1, 2, i));
  }
}

TEST(XtensorCompat_View, SingleNewaxisShape) {
  tt::tensor<double> a = {1, 2, 3, 4};

  auto v = tt::view(a, tt::newaxis);
  EXPECT_EQ(v.ndim(), 2u);
  EXPECT_EQ(v.shape()[0], 1u);
  EXPECT_EQ(v.shape()[1], 4u);
}

TEST(XtensorCompat_View, MultipleNewaxis) {
  tt::tensor<double> a = {1, 2, 3};

  auto v = tt::view(a, tt::newaxis, tt::all, tt::newaxis);
  EXPECT_EQ(v.ndim(), 3u);
  EXPECT_EQ(v.shape()[0], 1u);
  EXPECT_EQ(v.shape()[1], 3u);
  EXPECT_EQ(v.shape()[2], 1u);

  EXPECT_DOUBLE_EQ(v(0, 0, 0), 1);
  EXPECT_DOUBLE_EQ(v(0, 1, 0), 2);
  EXPECT_DOUBLE_EQ(v(0, 2, 0), 3);
}

// =============================================================================
// Complex Number Tests (adapted from test_xcomplex.cpp)
// =============================================================================

TEST(XtensorCompat_Complex, BasicComplex) {
  using cpx = std::complex<double>;
  tt::tensor<cpx> e = {{cpx(1.0, 0.0), cpx(1.0, 1.0)},
                       {cpx(1.0, -1.0), cpx(1.0, 0.0)}};

  EXPECT_EQ(e.ndim(), 2u);
  EXPECT_EQ(e.shape()[0], 2u);
  EXPECT_EQ(e.shape()[1], 2u);

  EXPECT_DOUBLE_EQ(e(0, 0).real(), 1.0);
  EXPECT_DOUBLE_EQ(e(0, 0).imag(), 0.0);
  EXPECT_DOUBLE_EQ(e(0, 1).real(), 1.0);
  EXPECT_DOUBLE_EQ(e(0, 1).imag(), 1.0);
  EXPECT_DOUBLE_EQ(e(1, 0).imag(), -1.0);
}

TEST(XtensorCompat_Complex, AbsComplex) {
  using cpx = std::complex<double>;
  tt::tensor<cpx> a = {cpx(3.0, 4.0), cpx(0.0, 1.0), cpx(1.0, 0.0)};

  auto result = tt::abs(a);

  // |3+4i| = 5
  EXPECT_DOUBLE_EQ(result(0), 5.0);
  // |0+1i| = 1
  EXPECT_DOUBLE_EQ(result(1), 1.0);
  // |1+0i| = 1
  EXPECT_DOUBLE_EQ(result(2), 1.0);
}

TEST(XtensorCompat_Complex, ComplexArithmetic) {
  using cpx = std::complex<double>;
  tt::tensor<cpx> a = {cpx(1.0, 2.0), cpx(3.0, 4.0)};
  tt::tensor<cpx> b = {cpx(1.0, -2.0), cpx(3.0, -4.0)};

  auto sum = a + b;
  EXPECT_DOUBLE_EQ(sum(0).real(), 2.0);
  EXPECT_DOUBLE_EQ(sum(0).imag(), 0.0);
  EXPECT_DOUBLE_EQ(sum(1).real(), 6.0);
  EXPECT_DOUBLE_EQ(sum(1).imag(), 0.0);

  auto prod = a * b;
  // (1+2i)(1-2i) = 1 + 4 = 5
  EXPECT_DOUBLE_EQ(prod(0).real(), 5.0);
  EXPECT_DOUBLE_EQ(prod(0).imag(), 0.0);
  // (3+4i)(3-4i) = 9 + 16 = 25
  EXPECT_DOUBLE_EQ(prod(1).real(), 25.0);
  EXPECT_DOUBLE_EQ(prod(1).imag(), 0.0);
}

TEST(XtensorCompat_Complex, ComplexScalarOps) {
  using cpx = std::complex<double>;
  tt::tensor<cpx> a = {cpx(1.0, 1.0), cpx(2.0, 2.0)};

  auto scaled = a * 2.0;
  EXPECT_DOUBLE_EQ(scaled(0).real(), 2.0);
  EXPECT_DOUBLE_EQ(scaled(0).imag(), 2.0);
  EXPECT_DOUBLE_EQ(scaled(1).real(), 4.0);
  EXPECT_DOUBLE_EQ(scaled(1).imag(), 4.0);
}

TEST(XtensorCompat_Complex, ComplexIteration) {
  using cpx = std::complex<double>;
  tt::tensor<cpx> a = {cpx(1.0, 0.0), cpx(0.0, 1.0), cpx(1.0, 1.0)};

  std::vector<cpx> collected;
  for (const auto& val : a) {
    collected.push_back(val);
  }

  EXPECT_EQ(collected.size(), 3u);
  EXPECT_EQ(collected[0], cpx(1.0, 0.0));
  EXPECT_EQ(collected[1], cpx(0.0, 1.0));
  EXPECT_EQ(collected[2], cpx(1.0, 1.0));
}

// =============================================================================
// Additional Builder/Manipulation Tests
// =============================================================================

TEST(XtensorCompat_Builder, ZerosLike) {
  tt::tensor<double> a = {{1, 2, 3}, {4, 5, 6}};
  auto z = tt::zeros<double>(a.shape());

  EXPECT_EQ(z.shape()[0], a.shape()[0]);
  EXPECT_EQ(z.shape()[1], a.shape()[1]);
  EXPECT_DOUBLE_EQ(z(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(z(1, 2), 0.0);
}

TEST(XtensorCompat_Builder, Full) {
  auto a = tt::full<int>(tt::shape_t{3, 4}, 42);

  EXPECT_EQ(a.ndim(), 2u);
  EXPECT_EQ(a.shape()[0], 3u);
  EXPECT_EQ(a.shape()[1], 4u);

  for (const auto& val : a) {
    EXPECT_EQ(val, 42);
  }
}

TEST(XtensorCompat_Manipulation, ReshapeChain) {
  tt::tensor<int> a = tt::arange(0, 24);

  a.reshape(tt::shape_t{2, 3, 4});
  EXPECT_EQ(a.ndim(), 3u);
  EXPECT_EQ(a.shape()[0], 2u);
  EXPECT_EQ(a.shape()[1], 3u);
  EXPECT_EQ(a.shape()[2], 4u);

  a.reshape(tt::shape_t{4, 6});
  EXPECT_EQ(a.ndim(), 2u);
  EXPECT_EQ(a.shape()[0], 4u);
  EXPECT_EQ(a.shape()[1], 6u);

  a.flatten();
  EXPECT_EQ(a.ndim(), 1u);
  EXPECT_EQ(a.shape()[0], 24u);
}

TEST(XtensorCompat_Manipulation, SqueezeEdgeCases) {
  // All dimensions are 1
  tt::tensor<int> a(tt::shape_t{1, 1, 1}, 5);
  a.squeeze();
  EXPECT_EQ(a.ndim(), 1u);
  EXPECT_EQ(a.shape()[0], 1u);
  EXPECT_EQ(a(0), 5);

  // No dimensions of size 1
  tt::tensor<int> b = {{1, 2}, {3, 4}};
  auto orig_shape = b.shape();
  b.squeeze();
  EXPECT_EQ(b.shape(), orig_shape);
}

TEST(XtensorCompat_Manipulation, Transpose1D) {
  tt::tensor<int> a = {1, 2, 3, 4, 5};
  auto tr = a.transposed();

  // 1D transpose should be identity
  EXPECT_EQ(tr.ndim(), 1u);
  EXPECT_EQ(tr.shape()[0], 5u);
  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(tr(i), a(i));
  }
}

TEST(XtensorCompat_Manipulation, Transpose3D) {
  tt::tensor<int> a(tt::shape_t{2, 3, 4});
  std::iota(a.begin(), a.end(), 0);

  auto tr = a.transposed();

  // Shape should be reversed
  EXPECT_EQ(tr.shape()[0], 4u);
  EXPECT_EQ(tr.shape()[1], 3u);
  EXPECT_EQ(tr.shape()[2], 2u);

  // Check transposition is correct
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 4; ++k) {
        EXPECT_EQ(a(i, j, k), tr(k, j, i));
      }
    }
  }
}

// =============================================================================
// Additional Math Tests
// =============================================================================

TEST(XtensorCompat_Math, Floor) {
  tt::tensor<double> a = {1.2, 2.7, -1.5, -2.8};
  auto result = tt::floor(a);

  EXPECT_DOUBLE_EQ(result(0), 1.0);
  EXPECT_DOUBLE_EQ(result(1), 2.0);
  EXPECT_DOUBLE_EQ(result(2), -2.0);
  EXPECT_DOUBLE_EQ(result(3), -3.0);
}

TEST(XtensorCompat_Math, Ceil) {
  tt::tensor<double> a = {1.2, 2.7, -1.5, -2.8};
  auto result = tt::ceil(a);

  EXPECT_DOUBLE_EQ(result(0), 2.0);
  EXPECT_DOUBLE_EQ(result(1), 3.0);
  EXPECT_DOUBLE_EQ(result(2), -1.0);
  EXPECT_DOUBLE_EQ(result(3), -2.0);
}

TEST(XtensorCompat_Math, Round) {
  tt::tensor<double> a = {1.2, 2.7, -1.5, -2.8, 2.5};
  auto result = tt::round(a);

  EXPECT_DOUBLE_EQ(result(0), 1.0);
  EXPECT_DOUBLE_EQ(result(1), 3.0);
  EXPECT_DOUBLE_EQ(result(2), -2.0);
  EXPECT_DOUBLE_EQ(result(3), -3.0);
  EXPECT_DOUBLE_EQ(result(4), 3.0);  // Banker's rounding or regular rounding
}

TEST(XtensorCompat_Math, Log10Log2) {
  tt::tensor<double> a = {1.0, 10.0, 100.0, 1000.0};
  auto log10_result = tt::log10(a);

  EXPECT_NEAR(log10_result(0), 0.0, 1e-10);
  EXPECT_NEAR(log10_result(1), 1.0, 1e-10);
  EXPECT_NEAR(log10_result(2), 2.0, 1e-10);
  EXPECT_NEAR(log10_result(3), 3.0, 1e-10);

  tt::tensor<double> b = {1.0, 2.0, 4.0, 8.0};
  auto log2_result = tt::log2(b);

  EXPECT_NEAR(log2_result(0), 0.0, 1e-10);
  EXPECT_NEAR(log2_result(1), 1.0, 1e-10);
  EXPECT_NEAR(log2_result(2), 2.0, 1e-10);
  EXPECT_NEAR(log2_result(3), 3.0, 1e-10);
}

TEST(XtensorCompat_Math, InverseTrig) {
  tt::tensor<double> a = {0.0, 0.5, 1.0};

  auto asin_result = tt::asin(a);
  EXPECT_NEAR(asin_result(0), 0.0, 1e-10);
  EXPECT_NEAR(asin_result(1), M_PI / 6, 1e-10);
  EXPECT_NEAR(asin_result(2), M_PI / 2, 1e-10);

  auto acos_result = tt::acos(a);
  EXPECT_NEAR(acos_result(0), M_PI / 2, 1e-10);
  EXPECT_NEAR(acos_result(1), M_PI / 3, 1e-10);
  EXPECT_NEAR(acos_result(2), 0.0, 1e-10);

  tt::tensor<double> b = {0.0, 1.0};
  auto atan_result = tt::atan(b);
  EXPECT_NEAR(atan_result(0), 0.0, 1e-10);
  EXPECT_NEAR(atan_result(1), M_PI / 4, 1e-10);
}

TEST(XtensorCompat_Math, Hyperbolic) {
  tt::tensor<double> a = {0.0, 1.0};

  auto sinh_result = tt::sinh(a);
  EXPECT_NEAR(sinh_result(0), 0.0, 1e-10);
  EXPECT_NEAR(sinh_result(1), std::sinh(1.0), 1e-10);

  auto cosh_result = tt::cosh(a);
  EXPECT_NEAR(cosh_result(0), 1.0, 1e-10);
  EXPECT_NEAR(cosh_result(1), std::cosh(1.0), 1e-10);

  auto tanh_result = tt::tanh(a);
  EXPECT_NEAR(tanh_result(0), 0.0, 1e-10);
  EXPECT_NEAR(tanh_result(1), std::tanh(1.0), 1e-10);
}

}  // namespace
