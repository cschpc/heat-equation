#include "field.hpp"
#include "io.hpp"
#include "parallel.hpp"
#include "utilities.hpp"
#include <iostream>

namespace heat {
Field initialize(const Input &input, const ParallelData &parallel) {
    auto [num_rows_global, num_cols_global, data] =
        input.read_file ? read_field(input.fname)
                        : generate_field(input.rows, input.cols);

    if (0 == parallel.rank) {
        std::cout << "Simulation parameters: "
                  << "rows: " << num_rows_global
                  << " columns: " << num_cols_global
                  << " time steps: " << input.nsteps << std::endl;
        std::cout << "Number of MPI tasks: " << parallel.size << std::endl;
    }

    auto [num_rows, num_cols] = Field::partition_domain(
        num_rows_global, num_cols_global, parallel.size);

    return Field(scatter(std::move(data), num_rows * num_cols), num_rows,
                 num_cols);
}
} // namespace heat
