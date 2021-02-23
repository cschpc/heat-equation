/* Heat equation solver in 2D. */

#include <string>
#include <iostream>
#include <stdio.h>
#include <mpi.h>

#include "heat.hpp"

int main(int argc, char **argv)
{

    int const image_interval = 100;    // Image output interval

    MPI_Init(&argc, &argv);
    ParallelData parallelization; // Parallelization info

    int nsteps;                 // Number of time steps
    Field current, previous;    // Current and previous temperature fields
    initialize(argc, argv, current, previous, nsteps, parallelization);

    // Output the initial field
    // write_field(&current, 0, &parallelization);

    auto average_temp = average(current);
    if (parallelization.rank == 0) {
        std::cout << "Average temperature at start: " << average_temp << std::endl;
    }    

    // Largest stable time step 
    double const a = 0.5;     // Diffusion constant 
    auto dx2 = current.dx * current.dx;
    auto dy2 = current.dy * current.dy;
    auto dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    //Get the start time stamp 
    auto start_clock = MPI_Wtime();

    /* Time evolve */
    for (int iter = 1; iter <= nsteps; iter++) {
        exchange(previous, parallelization);
        evolve(current, previous, a, dt);
        if (iter % image_interval == 0) {
            write_field(current, iter, parallelization);
        }
        /* Swap current field so that it will be used
            as previous for next iteration step */
        current.swap(previous);
    }

    auto stop_clock = MPI_Wtime();

    /* Average temperature for reference */
    average_temp = average(previous);

    /* Determine the CPU time used for the iteration */
    if (parallelization.rank == 0) {
        printf("Iteration took %.3f seconds.\n", (stop_clock - start_clock));
        printf("Average temperature: %f\n", average_temp);
        if (argc == 1) {
            printf("Reference value with default arguments: 59.281239\n");
        }
    }

    /* Output the final field */
    // write_field(&previous, nsteps, &parallelization);

    // finalize(&current, &previous);
    MPI_Finalize();

    return 0;
}
