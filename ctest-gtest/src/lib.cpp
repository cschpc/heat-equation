/* Heat equation solver in 2D. */

#include <iomanip>
#include <iostream>
#include <mpi.h>

#include "constants.hpp"
#include "core.hpp"
#include "field.hpp"
#include "io.hpp"
#include "parallel.hpp"
#include "utilities.hpp"

namespace heat{
std::tuple<std::vector<double>, int, int>
initialize(const Input &input, const ParallelData &parallel) {
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

    return std::make_tuple(scatter(std::move(data), num_rows * num_cols),
                           num_rows, num_cols);
}

void run(std::string &&fname) {
    // Parallelization info
    ParallelData parallelization;
    const Input input = read_input(std::move(fname), parallelization.rank);
    const Constants constants(input);

    // Temperature fields
    auto [data, num_rows, num_cols] = initialize(input, parallelization);
    Field current(std::move(data), num_rows, num_cols);
    Field previous = current;

    // Output the initial field
    write_field(current, parallelization,
                make_png_filename(input.png_name_prefix.c_str(), 0));

    auto avg = average(current, parallelization);
    if (0 == parallelization.rank) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average temperature at start: " << avg << std::endl;
    }

    //Get the start time stamp
    const auto start_clock = MPI_Wtime();

    // Time evolve
    for (int iter = 1; iter <= input.nsteps; iter++) {
        exchange(previous, parallelization);
        evolve(current, previous, constants);

        if (iter % input.image_interval == 0) {
            write_field(current, parallelization,
                        make_png_filename(input.png_name_prefix.c_str(), iter));
        }

        // Swap current field so that it will be used
        // as previous for next iteration step
        current.swap(previous);
    }

    const auto stop_clock = MPI_Wtime();
    constexpr double ref_val = 59.763305;

    avg = average(previous, parallelization);
    if (0 == parallelization.rank) {
        std::cout << "Iteration took " << (stop_clock - start_clock)
                  << " seconds." << std::endl;
        std::cout << "Average temperature: " << avg << std::endl;

        if (not input.read_file) {
            std::cout << "Reference value with default arguments: " << ref_val
                      << std::endl;
        }
    }

    // Output the final field
    write_field(previous, parallelization,
                make_png_filename(input.png_name_prefix.c_str(), input.nsteps));
}
}
