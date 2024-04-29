#include "field.hpp"
#include "parallel.hpp"
#include "pngwriter.h"
#include "utilities.hpp"
#include <gtest/gtest.h>
#include <mpi.h>

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

TEST(field_test, generation_correct) {
    int argc = 1;
    char **argv = NULL;
    MPI_Init(&argc, &argv);
    ParallelData parallel;
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    Field field = {};
    field.setup(num_rows, num_cols, parallel);
    field.generate(parallel);

    std::vector<double> field_data =
        heat::generate_field(num_rows, num_cols, 0);
    save_png(field_data.data(), num_rows, num_cols, "field_data.png", 'c');
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            const int index = i * num_cols + j;
            ASSERT_EQ(field(i + 1, j + 1), field_data[index]);
        }
    }

    MPI_Finalize();
}
