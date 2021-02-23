#include "heat.hpp"
#include <iostream>
#include <mpi.h>

void Field::setup(int nx_in, int ny_in, ParallelData parallel) 
{
    nx_full = nx_in;
    ny_full = ny_in;

    nx = nx_full / parallel.size;
    if (nx * parallel.size != nx_full) {
        std::cout << "Cannot divide grid evenly to processors" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -2);
    }
    ny = ny_full;

   // size includes ghost layers
   std::size_t field_size = (nx + 2) * (ny + 2);

   temperature.reserve(field_size);
}

void Field::swap(Field& other)
{
    temperature.swap(other.temperature);
}

void Field::generate(ParallelData parallel) {

    int ind;
    double radius;
    int dx, dy;

    // Radius of the source disc 
    radius = nx_full / 6.0;
    for (int i = 0; i < nx + 2; i++) {
        for (int j = 0; j < ny + 2; j++) {
            ind = i * (ny + 2) + j;
            /* Distance of point i, j from the origin */
            dx = i + parallel.rank * nx - nx_full / 2 + 1;
            dy = j - ny / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                // data[ind] = 5.0;
                temperature.push_back(5.0);
            } else {
                // data[ind] = 65.0;
                temperature.push_back(65.0);
            }
        }
    }

    // Boundary conditions
    for (int i = 0; i < nx + 2; i++) {
        temperature[i * (ny + 2)] = 20.0;
        temperature[i * (ny + 2) + ny + 1] = 70.0;
    }

    if (parallel.rank == 0) {
        for (int j = 0; j < ny + 2; j++) {
            ind = j;
            temperature[ind] = 85.0;
        }
    }
    if (parallel.rank == parallel.size - 1) {
        for (int j = 0; j < ny + 2; j++) {
            ind = (nx + 1) * (ny + 2) + j;
            temperature[ind] = 5.0;
        }
    }
}



