#include "field.hpp"
#include "parallel.hpp"

#include <iostream>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <utility>

void Field::setup(int num_rows_in, int num_cols_in,
                  const ParallelData &parallel) {
    const auto [nr, nc] =
        Field::partition_domain(num_rows_in, num_cols_in, parallel.size);
    num_rows_global = num_rows_in;
    num_cols_global = num_cols_in;
    num_rows = nr;
    num_cols = nc;
    // matrix includes also ghost layers
    temperature = Matrix<double>(num_rows + 2, num_cols + 2);
}

void Field::generate(const ParallelData &parallel) {

    // Radius of the source disc
    auto radius = num_rows_global / 6.0;
    for (int i = 0; i < num_rows + 2; i++) {
        for (int j = 0; j < num_cols + 2; j++) {
            // Distance of point i, j from the origin
            auto dx = i + parallel.rank * num_rows - num_rows_global / 2 + 1;
            auto dy = j - num_cols / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                temperature(i, j) = 5.0;
            } else {
                temperature(i, j) = 65.0;
            }
        }
    }

    // Boundary conditions
    for (int i = 0; i < num_rows + 2; i++) {
        // Left
        temperature(i, 0) = 20.0;
        // Right
        temperature(i, num_cols + 1) = 70.0;
    }

    // Top
    if (0 == parallel.rank) {
        for (int j = 0; j < num_cols + 2; j++) {
            temperature(0, j) = 85.0;
        }
    }
    // Bottom
    if (parallel.rank == parallel.size - 1) {
        for (int j = 0; j < num_cols + 2; j++) {
            temperature(num_rows + 1, j) = 5.0;
        }
    }
}

std::pair<int, int> Field::partition_domain(int num_rows, int num_cols,
                                            int num_partitions) {
    const int nr = num_rows / num_partitions;
    if (nr * num_partitions != num_rows) {
        std::stringstream ss;
        ss << "Could not partition " << num_rows << " rows and " << num_cols
           << " columns evenly to " << num_partitions << " partitions";
        throw std::runtime_error(ss.str());
    }
    // Columns are not partitioned
    const int nc = num_cols;

    return std::make_pair(nr, nc);
}
