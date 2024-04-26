// Utility functions for heat equation solver
// NOTE: This file does not need to be edited! 

#include <mpi.h>

#include "field.hpp"
#include "parallel.hpp"

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
