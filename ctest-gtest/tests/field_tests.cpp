#include "field.hpp"
#include "parallel.hpp"
#include "utilities.hpp"

#include <gtest/gtest.h>
#include <mpi.h>
#include <numeric>

TEST(field_test, domain_partition_succeeds) {
    constexpr int num_rows = 100;
    constexpr int num_cols = 100;
    constexpr int num_partitions = 10;
    EXPECT_NO_THROW(
        Field::partition_domain(num_rows, num_cols, num_partitions));
}

TEST(field_test, domain_partition_throws_an_exception) {
    constexpr int num_rows = 101;
    constexpr int num_cols = 100;
    constexpr int num_partitions = 10;
    EXPECT_THROW(
        {
            try {
                Field::partition_domain(num_rows, num_cols, num_partitions);
            } catch (const std::runtime_error &e) {
                EXPECT_STREQ("Could not partition 101 rows and 100 columns "
                             "evenly to 10 partitions",
                             e.what());
                throw;
            }
        },
        std::runtime_error);
}

TEST(field_test, field_construction) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    std::vector<double> v(num_rows * num_cols);
    std::iota(v.begin(), v.end(), 0.0);
    const Field field(std::move(v), num_rows, num_cols);

    ASSERT_EQ(field.num_rows, num_rows);
    ASSERT_EQ(field.num_cols, num_cols);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            const double value = static_cast<double>(i * num_cols + j);
            // Field doesn't sample the ghost layers
            ASSERT_EQ(field(i, j), value);
        }
    }
}
