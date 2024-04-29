#include <gtest/gtest.h>

#include "utilities.hpp"

TEST(utilities_test, data_generated_correctly) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    const std::vector<double> data =
        heat::generate_field(num_rows, num_cols, 0);

    ASSERT_EQ(num_rows * num_cols, data.size());

    for (int i = 0; i < num_cols; i++) {
        ASSERT_EQ(data[i], 65.0)
            << "First row of generated data should contain 65.0";
    }
}
