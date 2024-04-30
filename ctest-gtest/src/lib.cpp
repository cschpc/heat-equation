/* Heat equation solver in 2D. */

#include <iomanip>
#include <iostream>
#include <mpi.h>

#include "field.hpp"
#include "heat.hpp"
#include "io.hpp"
#include "parallel.hpp"
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
    write_field(current, 0, parallelization);

    auto avg = heat::average(current, parallelization);
    if (0 == parallelization.rank) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average temperature at start: " << avg << std::endl;
    }

    // Diffusion constant
    constexpr double a = 0.5;
    constexpr double dx2 = Field::dx2;
    constexpr double dy2 = Field::dy2;

    // Largest stable time step
    constexpr double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    //Get the start time stamp
    const auto start_clock = MPI_Wtime();

    // Time evolve
    for (int iter = 1; iter <= input.nsteps; iter++) {
        exchange(previous, parallelization);
        evolve(current, previous, a, dt);

        if (iter % input.image_interval == 0) {
            write_field(current, iter, parallelization);
        }

        // Swap current field so that it will be used
        // as previous for next iteration step
        std::swap(current, previous);
    }

    const auto stop_clock = MPI_Wtime();
    constexpr double ref_val = 59.075405;

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
    write_field(previous, input.nsteps, parallelization);
}
}
