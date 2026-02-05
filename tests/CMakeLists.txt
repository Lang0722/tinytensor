# Find Google Test
find_package(GTest REQUIRED)

# Test executable for tensor tests
add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test
    PRIVATE
        tinytensor::tinytensor
        GTest::gtest
        GTest::gtest_main
)

# Test executable for xtensor compatibility tests
add_executable(xtensor_compat_test xtensor_compat_test.cpp)
target_link_libraries(xtensor_compat_test
    PRIVATE
        tinytensor::tinytensor
        GTest::gtest
        GTest::gtest_main
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(tensor_test)
gtest_discover_tests(xtensor_compat_test)
