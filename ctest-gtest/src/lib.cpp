/* Heat equation solver in 2D. */

#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "heat.hpp"

namespace heat{
void run(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // Image output interval
    constexpr int image_interval = 1500;

    // Parallelization info
    ParallelData parallelization = {};

    // Number of time steps
    int nsteps = 0;

    // Temperature fields
    Field current = {};
    Field previous = {};
    initialize(argc, argv, current, previous, nsteps, parallelization);

    // Output the initial field
    write_field(current, 0, parallelization);

    auto avg = average(current);
    if (0 == parallelization.rank) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average temperature at start: " << avg << std::endl;
    }

    // Diffusion constant
    constexpr double a = 0.5;
    const auto dx2 = current.dx * current.dx;
    const auto dy2 = current.dy * current.dy;

    // Largest stable time step
    const auto dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    //Get the start time stamp
    const auto start_clock = MPI_Wtime();

    // Time evolve
    for (int iter = 1; iter <= nsteps; iter++) {
        exchange(previous, parallelization);
        evolve(current, previous, a, dt);

        if (iter % image_interval == 0) {
            write_field(current, iter, parallelization);
        }

        // Swap current field so that it will be used
        // as previous for next iteration step
        std::swap(current, previous);
    }

    const auto stop_clock = MPI_Wtime();
    constexpr auto ref_val = 59.281239;

    avg = average(previous);
    if (0 == parallelization.rank) {
        std::cout << "Iteration took " << (stop_clock - start_clock)
                  << " seconds." << std::endl;
        std::cout << "Average temperature: " << avg << std::endl;

        if (1 == argc) {
            std::cout << "Reference value with default arguments: " << ref_val
                      << std::endl;
        }
    }

    // Output the final field
    write_field(previous, nsteps, parallelization);

    MPI_Finalize();
}
}

