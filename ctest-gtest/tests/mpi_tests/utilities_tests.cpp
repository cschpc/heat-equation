#include <gtest/gtest.h>

#include "field.hpp"
#include "parallel.hpp"
#include "utilities.hpp"

// TODO:
// - gather/scatter
// - proper multi process average
TEST(utilities_test, zero_field_average_is_zero) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    const Field field(std::vector<double>(num_rows * num_cols), num_rows,
                      num_cols);
    ParallelData pd;
    ASSERT_EQ(heat::average(field, pd), 0.0);
}

TEST(utilities_test, unity_field_average_is_one) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 100;
    const Field field(std::vector<double>(num_rows * num_cols, 1.0), num_rows,
                      num_cols);
    ParallelData pd;
    ASSERT_EQ(heat::average(field, pd), 1.0);
}
