#include <gtest/gtest.h>
#include <mpi.h>

#include "field.hpp"
#include "io.hpp"
#include "parallel.hpp"
#include "utilities.hpp"

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
