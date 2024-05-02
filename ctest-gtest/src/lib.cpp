/* Heat equation solver in 2D. */

#include <iomanip>
#include <iostream>
#include <mpi.h>

#include "core.hpp"
#include "field.hpp"
#include "io.hpp"
#include "parallel.hpp"
#include "setup.hpp"
#include "utilities.hpp"

namespace heat{
void run(std::string &&fname) {
    // Parallelization info
    ParallelData parallelization;
    const Input input = read_input(std::move(fname), parallelization.rank);

    // Temperature fields
    Field current = initialize(input, parallelization);
    Field previous = current;

    // Output the initial field
    heat::write_field(
        current, parallelization,
        heat::make_png_filename(input.png_name_prefix.c_str(), 0));

    auto avg = heat::average(current, parallelization);
    if (0 == parallelization.rank) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average temperature at start: " << avg << std::endl;
    }

    constexpr double dx2 = Field::dx2;
    constexpr double dy2 = Field::dy2;

    // Largest stable time step
    const double dt =
        dx2 * dy2 / (2.0 * input.diffusion_constant * (dx2 + dy2));

    //Get the start time stamp
    const auto start_clock = MPI_Wtime();

    // Time evolve
    for (int iter = 1; iter <= input.nsteps; iter++) {
        exchange(previous, parallelization);
        evolve(current, previous, input.diffusion_constant, dt);

        if (iter % input.image_interval == 0) {
            heat::write_field(
                current, parallelization,
                heat::make_png_filename(input.png_name_prefix.c_str(), iter));
        }

        // Swap current field so that it will be used
        // as previous for next iteration step
        std::swap(current, previous);
    }

    const auto stop_clock = MPI_Wtime();
    constexpr double ref_val = 59.763305;

    avg = heat::average(previous, parallelization);
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
    heat::write_field(
        previous, parallelization,
        heat::make_png_filename(input.png_name_prefix.c_str(), input.nsteps));
}
}
