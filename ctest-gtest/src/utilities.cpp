// Utility functions for heat equation solver
// NOTE: This file does not need to be edited! 

#include <mpi.h>
#include <stdexcept>

#include "field.hpp"
#include "parallel.hpp"
#include "utilities.hpp"

namespace heat {
// Calculate average temperature
double average(const Field &field, const ParallelData &parallel) {
    double local_average = 0.0;
    double average = 0.0;

    for (int i = 0; i < field.num_rows; i++) {
        for (int j = 0; j < field.num_cols; j++) {
            local_average += field(i, j);
        }
    }

    MPI_Allreduce(&local_average, &average, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    average /= (field.num_rows * field.num_cols * parallel.size);
    return average;
}

std::vector<double> generate_field(int num_rows, int num_cols, int rank) {
    std::vector<double> data;
    if (rank == 0) {
        data.resize(num_rows * num_cols);
        // Radius of the source disc
        const auto radius = num_rows / 6.0;
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < num_cols; j++) {
                const auto index = i * num_cols + j;
                // Distance of point i, j from the origin
                const auto dx = i - num_rows / 2 + 2;
                const auto dy = j - num_cols / 2 + 2;
                data[index] = dx * dx + dy * dy < radius * radius ? 5.0 : 65.0;
            }
        }
    }
    return data;
}

std::vector<double> scatter(std::vector<double> &&full_data,
                            int num_values_per_rank, int n) {
    std::vector<double> my_data(num_values_per_rank);
    MPI_Scatter(full_data.data(), num_values_per_rank, MPI_DOUBLE,
                my_data.data(), num_values_per_rank, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    return my_data;
}
} // namespace heat
