#include "field.hpp"
#include "parallel.hpp"

#include <iostream>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <utility>

Field::Field(std::vector<double> &&data, int num_rows, int num_cols)
    : num_rows(num_rows), num_cols(num_cols),
      temperature(Matrix<double>::make_with_ghost_layers(std::move(data),
                                                         num_rows, num_cols)) {}

double Field::sum() const {
    double sum = 0.0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            sum += (*this)(i, j);
        }
    }

    return sum;
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
