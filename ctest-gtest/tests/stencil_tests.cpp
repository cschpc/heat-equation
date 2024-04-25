#include "heat.hpp"
#include <gtest/gtest.h>

TEST(stencil_test, zero_field_gives_zero) {
    Field field = {};
    field.setup(1, 1, ParallelData());
    constexpr double a = 1.0;
    constexpr double dt = 1.0;
    constexpr int i = 1;
    constexpr int j = 1;

    const double value = heat::stencil(i, j, field, a, dt);
    ASSERT_EQ(value, 0.0) << "Value should be zero";
}
