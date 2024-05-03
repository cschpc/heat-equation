#include "field.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <utility>

Field::Field(std::vector<double> &&data, int num_rows, int num_cols)
    : num_rows(num_rows), num_cols(num_cols),
      temperatures((num_rows + 2) * (num_cols + 2)) {
    // Copy the real data to the inner part
    for (int i = 0; i < num_rows; i++) {
        const int row = i + 1;
        const int width = num_cols + 2;
        constexpr int column = 1;
        const int offset = row * width + column;
        auto from = data.begin() + i * num_cols;
        auto to = temperatures.begin() + offset;
        std::copy_n(from, num_cols, to);
    }

    // Make the ghost layers
    const int nr = num_rows + 2;
    const int nc = num_cols + 2;
    for (int i = 0; i < nr; i++) {
        const int first = i * nc;
        const int last = (i + 1) * nc - 1;
        // Left
        temperatures[first] = temperatures[first + 1];
        // Right
        temperatures[last] = temperatures[last - 1];
    }

    for (int j = 0; j < nc; j++) {
        const int first = j;
        const int last = (nr - 1) * nc + j;
        // Top
        temperatures[first] = temperatures[first + nc];
        // Bottom
        temperatures[last] = temperatures[last - nc];
    }
}

double Field::sum() const {
    double sum = 0.0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            sum += (*this)(i, j);
        }
    }

    return sum;
}

std::vector<double> Field::get_temperatures() const {
    // Copy the "real" data, skip the ghost layers on top, bottom, left and
    // right
    std::vector<double> data;
    data.reserve(num_rows * num_cols);

    for (int i = 0; i < num_rows; i++) {
        const int row = i + 1;
        const int width = num_cols + 2;
        constexpr int column = 1;
        const int offset = row * width + column;
        auto from = temperatures.begin() + offset;
        std::copy_n(from, num_cols, std::back_inserter(data));
    }

    return data;
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
