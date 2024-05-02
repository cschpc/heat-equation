#include "field.hpp"
#include "parallel.hpp"
#include "utilities.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

TEST(utilities_test, data_generated_correctly) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    const std::vector<double> data =
        std::get<2>(heat::generate_field(num_rows, num_cols));

    ASSERT_EQ(num_rows * num_cols, data.size());

    for (int i = 0; i < num_cols; i++) {
        ASSERT_EQ(data[i], 65.0)
            << "First row of generated data should contain 65.0";
    }
}

TEST(utilities_test, zero_field_average_is_zero) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    const Field field(std::vector<double>(num_rows * num_cols), num_rows,
                      num_cols);
    int argc = 1;
    char **argv = {};
    MPI_Init(&argc, &argv);
    ParallelData pd;
    ASSERT_EQ(heat::average(field, pd), 0.0);
    MPI_Finalize();
}

TEST(utilities_test, unity_field_average_is_one) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 100;
    const Field field(std::vector<double>(num_rows * num_cols, 1.0), num_rows,
                      num_cols);
    int argc = 1;
    char **argv = {};
    MPI_Init(&argc, &argv);
    ParallelData pd;
    ASSERT_EQ(heat::average(field, pd), 1.0);
    MPI_Finalize();
}
