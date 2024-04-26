#include "heat.hpp"
#include <gtest/gtest.h>

TEST(field_test, domain_partition_succeeds) {
    const int width = 100;
    const int height = 100;
    const int num_partitions = 10;
    EXPECT_NO_THROW(Field::partition_domain(width, height, num_partitions));
}

TEST(field_test, domain_partition_throws_an_exception) {
    const int width = 101;
    const int height = 100;
    const int num_partitions = 10;
    EXPECT_THROW(
        {
            try {
                Field::partition_domain(width, height, num_partitions);
            } catch (const std::runtime_error &e) {
                EXPECT_STREQ("Could not partition width (101) and height (100) "
                             "evenly to 10 partitions",
                             e.what());
                throw;
            }
        },
        std::runtime_error);
}
