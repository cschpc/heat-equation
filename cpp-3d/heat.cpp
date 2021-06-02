#include "heat.hpp"
#include "matrix.hpp"
#include <iostream>
#include <mpi.h>

void Field::setup(int nx_in, int ny_in, int nz_in, ParallelData parallel) 
{
    nx_full = nx_in;
    ny_full = ny_in;
    nz_full = nz_in;

    nx = nx_full / parallel.size;
    if (nx * parallel.size != nx_full) {
        std::cout << "Cannot divide grid evenly to processors" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -2);
    }
    ny = ny_full;
    nz = nz_full;

   // matrix includes also ghost layers
   temperature = Matrix<double> (nx + 2, ny + 2, nz + 2);
}

void Field::generate(ParallelData parallel) {

    // Radius of the source disc 
    auto radius = nx_full / 6.0;
    for (int i = 0; i < nx + 2; i++) {
        for (int j = 0; j < ny + 2; j++) {
            for (int k = 0; k < nz + 2; k++) {
                // Distance of point i, j, k from the origin 
                auto dx = i + parallel.rank * nx - nx_full / 2 + 1;
                auto dy = j - ny / 2 + 1;
                auto dz = k - nz / 2 + 1;
                if (dx * dx + dy * dy + dz * dz < radius * radius) {
                    temperature(i, j, k) = 5.0;
                } else {
                    temperature(i, j, k) = 65.0;
                }
            }
        }
    }

    // Boundary conditions
    for (int i = 0; i < nx + 2; i++) {
      for (int j = 0; j < ny + 2; j++) {
        // Left
        temperature(i, j, 0) = 20.0;
        // Right
        temperature(i, j, nz + 1) = 70.0;
      }
    }

    for (int i = 0; i < nx + 2; i++) {
      for (int k = 0; k < nz + 2; k++) {
        // Left
        temperature(i, 0, k) = 33.0;
        // Right
        temperature(i, ny + 1, k) = 55.0;
      }
    }

    // Top
    if (0 == parallel.rank) {
        for (int j = 0; j < ny + 2; j++) {
          for (int k = 0; k < nz + 2; k++) {
            temperature(0, j, k) = 85.0;
          }
        }
    }
    // Bottom
    if (parallel.rank == parallel.size - 1) {
        for (int j = 0; j < ny + 2; j++) {
          for (int k = 0; k < nz + 2; k++) {
            temperature(nx + 1, j, k) = 5.0;
          }
        }
    }
}
