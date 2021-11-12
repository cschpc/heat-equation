/*****************************************************************************
MIT License

Copyright (c) 2021 CSC HPC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

// Simple 3d heat equation solver

#include <string>
#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "heat.hpp"

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    const int image_interval = 1500;    // Image output interval

    ParallelData parallelization; // Parallelization info

    int nsteps;                 // Number of time steps
    Field current, previous;    // Current and previous temperature fields
    initialize(argc, argv, current, previous, nsteps, parallelization);

    // Output the initial field
    write_field(current, 0, parallelization);

    auto average_temp = average(current);
    if (0 == parallelization.rank) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average temperature at start: " << average_temp << std::endl;
    }    

    const double a = 0.5;     // Diffusion constant 
    auto dx2 = current.dx * current.dx;
    auto dy2 = current.dy * current.dy;
    auto dz2 = current.dz * current.dz;
    // Largest stable time step 
    auto dt = dx2 * dy2 * dz2 / (2.0 * a * (dx2 + dy2 + dz2));

    //Get the start time stamp 
    auto start_clock = MPI_Wtime();

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

    auto stop_clock = MPI_Wtime();

    // Average temperature for reference 
    average_temp = average(previous);

    if (0 == parallelization.rank) {
        std::cout << "Iteration took " << (stop_clock - start_clock)
                  << " seconds." << std::endl;
        std::cout << "Average temperature: " << average_temp << std::endl;
        if (1 == argc) {
            std::cout << "Reference value with default arguments: " 
                      << 63.834383 << std::endl;
        }
    }

    // Output the final field
    write_field(previous, nsteps, parallelization);

    MPI_Finalize();

    return 0;
}
