// Copyright (c) 2021 CSC HPC
// SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

// Simple 3d heat equation solver

#include <string>
#include <iostream>
#include <iomanip>
#ifndef NO_MPI
#include <mpi.h>
#endif
#include "parallel.hpp"
#include "heat.hpp"
#include "functions.hpp"

int main(int argc, char **argv)
{

#ifndef NO_MPI
    int provided;
    int required = MPI_THREAD_SERIALIZED;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required) {
       std::cout << "No required MPI thread safety support" << std::endl;
       MPI_Abort(MPI_COMM_WORLD, -1);
    }
#endif

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
    auto start_clock = timer();
    double start_mpi, start_comp;
    double t_mpi = 0.0;
    double t_comp = 0.0;
 #pragma omp parallel reduction(+:t_comp)
  {
    // Time evolve
    for (int iter = 1; iter <= nsteps; iter++) {
        #pragma omp single
        {
         start_mpi = timer();
         exchange(previous, parallelization);
         t_mpi += timer() - start_mpi;
        }
        start_comp = timer();
        evolve(current, previous, a, dt);
        t_comp += timer() - start_comp;
        if (iter % image_interval == 0) {
            #pragma omp single
            write_field(current, iter, parallelization);
        }
        // Swap current field so that it will be used
        // as previous for next iteration step
        #pragma omp single
        std::swap(current, previous);
    }
  } // end omp parallel

    auto stop_clock = timer();
    t_comp /= parallelization.num_threads;

    // Average temperature for reference 
    average_temp = average(previous);

    if (0 == parallelization.rank) {
        std::cout << "Iteration took " << (stop_clock - start_clock)
                  << " seconds." << std::endl;
        std::cout << "  MPI           " << t_mpi << " s." << std::endl;
        std::cout << "  Compute       " << t_comp << " s." << std::endl;
        std::cout << "Average temperature: " << average_temp << std::endl;
        if (1 == argc) {
            std::cout << "Reference value with default arguments: " 
                      << 63.834223 << std::endl;
        }
    }

    // Output the final field
    write_field(previous, nsteps, parallelization);

#ifndef NO_MPI
    parallelization.finalize();
    MPI_Finalize();
#endif

    return 0;
}
