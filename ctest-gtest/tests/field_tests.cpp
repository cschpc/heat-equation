#include "field.hpp"
#include <gtest/gtest.h>

TEST(field_test, domain_partition_succeeds) {
    const int num_rows = 100;
    const int num_cols = 100;
    const int num_partitions = 10;
    EXPECT_NO_THROW(
        Field::partition_domain(num_rows, num_cols, num_partitions));
}

TEST(field_test, domain_partition_throws_an_exception) {
    const int num_rows = 101;
    const int num_cols = 100;
    const int num_partitions = 10;
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
