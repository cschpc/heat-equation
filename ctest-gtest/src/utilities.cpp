// Utility functions for heat equation solver
// NOTE: This file does not need to be edited! 

#include <mpi.h>

#include "field.hpp"
#include "parallel.hpp"
#include "utilities.hpp"

namespace heat {
// Calculate average temperature
double average(const Field &field, const ParallelData &parallel) {
    double local_average = 0.0;
    double average = 0.0;

    for (int i = 1; i < field.num_rows + 1; i++) {
        for (int j = 1; j < field.num_cols + 1; j++) {
            local_average += field.temperature(i, j);
        }
    }

    MPI_Allreduce(&local_average, &average, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    average /= (field.num_rows * field.num_cols * parallel.size);
    return average;
}

std::vector<double> generate_field(int num_rows, int num_cols) {
    std::vector<double> data(num_rows * num_cols);
    // Radius of the source disc
    const auto radius = num_rows / 6.0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            const auto index = i * num_cols + j;
            // Distance of point i, j from the origin
            const auto dx = i - num_rows / 2 + 1;
            const auto dy = j - num_cols / 2 + 1;
            data[index] = dx * dx + dy * dy < radius * radius ? 5.0 : 65.0;
        }
    }

    return data;
}
} // namespace heat
