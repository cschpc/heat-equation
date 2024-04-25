/* Heat equation solver in 2D. */

#include <iomanip>
#include <iostream>
#include <mpi.h>

#include "heat.hpp"
#include "input.hpp"

namespace heat{
void run(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // Parallelization info
    ParallelData parallelization;

    // Read input from file
    std::string fname = "";
    if (argc > 1) {
        fname = argv[1];
    }
    const Input input = read_input(fname.c_str());

    // Temperature fields
    Field current = {};
    Field previous = {};
    initialize(input, current, previous, parallelization);

    // Output the initial field
    write_field(current, 0, parallelization);

    auto avg = average(current);
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
    constexpr double ref_val = 59.281239;

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
    write_field(previous, input.nsteps, parallelization);

    MPI_Finalize();
}
}
