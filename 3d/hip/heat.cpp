#include "heat.hpp"
#include "parallel.hpp"
#include "matrix.hpp"
#include <iostream>
#include <mpi.h>

void Field::setup(int nx_in, int ny_in, int nz_in, ParallelData& parallel) 
{
    nx_full = nx_in;
    ny_full = ny_in;
    nz_full = nz_in;

    int dims[3], periods[3], coords[3];
    MPI_Cart_get(parallel.comm, 3, dims, periods, coords);

    nx = nx_full / dims[0];
    if (nx * dims[0] != nx_full) {
      std::cout << "Cannot divide grid evenly to processors" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -2);
    }
    ny = ny_full / dims[1];
    if (ny * dims[1] != ny_full) {
      std::cout << "Cannot divide grid evenly to processors" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -2);
    }

    nz = nz_full / dims[2];
    if (nz * dims[2] != nz_full) {
      std::cout << "Cannot divide grid evenly to processors" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -2);
    }

    // matrix includes also ghost layers
    temperature = Matrix<double> (nx + 2, ny + 2, nz + 2);

    // Communication buffers / datatypes
    int sizes[3];
    int offsets[3] = {0, 0, 0};
    int subsizes[3];
#ifdef MPI_DATATYPES
    sizes[0] = nx + 2;
    sizes[1] = ny + 2;
    sizes[2] = nz + 2;
    subsizes[0] = 1;
    subsizes[1] = ny + 2;
    subsizes[2] = nz + 2;
    MPI_Type_create_subarray(3, sizes, subsizes, offsets, MPI_ORDER_C,
                             MPI_DOUBLE, &parallel.halotypes[0]);
    MPI_Type_commit(&parallel.halotypes[0]);
    subsizes[0] = nx + 2;
    subsizes[1] = 1;
    subsizes[2] = nz + 2;
    MPI_Type_create_subarray(3, sizes, subsizes, offsets, MPI_ORDER_C,
                             MPI_DOUBLE, &parallel.halotypes[1]);
    MPI_Type_commit(&parallel.halotypes[1]);
    subsizes[0] = nx + 2;
    subsizes[1] = ny + 2;
    subsizes[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, offsets, MPI_ORDER_C,
                             MPI_DOUBLE, &parallel.halotypes[2]);
    MPI_Type_commit(&parallel.halotypes[2]);
#else
    parallel.send_buffers[0][0] = Matrix<double> (ny + 2, nz + 2);
    parallel.send_buffers[0][1] = Matrix<double> (ny + 2, nz + 2);
    parallel.send_buffers[1][0] = Matrix<double> (nx + 2, nz + 2);
    parallel.send_buffers[1][1] = Matrix<double> (nx + 2, nz + 2);
    parallel.send_buffers[2][0] = Matrix<double> (nx + 2, ny + 2);
    parallel.send_buffers[2][1] = Matrix<double> (nx + 2, ny + 2);
    parallel.recv_buffers[0][0] = Matrix<double> (ny + 2, nz + 2);
    parallel.recv_buffers[0][1] = Matrix<double> (ny + 2, nz + 2);
    parallel.recv_buffers[1][0] = Matrix<double> (nx + 2, nz + 2);
    parallel.recv_buffers[1][1] = Matrix<double> (nx + 2, nz + 2);
    parallel.recv_buffers[2][0] = Matrix<double> (nx + 2, ny + 2);
    parallel.recv_buffers[2][1] = Matrix<double> (nx + 2, ny + 2);
#endif

    // MPI datatype for subblock needed in I/O
    // Rank 0 uses datatype for receiving data into full array while
    // other ranks use datatype for sending the inner part of array
    subsizes[0] = nx;
    subsizes[1] = ny;
    subsizes[2] = nz;
    if (parallel.rank == 0) {
        sizes[0] = nx_full;
        sizes[1] = ny_full;
        sizes[2] = nz_full;
    } else {
        sizes[0] = nx + 2;
        sizes[1] = ny + 2;
        sizes[2] = nz + 2;
    }
    MPI_Type_create_subarray(3, sizes, subsizes, offsets, MPI_ORDER_C,
                             MPI_DOUBLE, &parallel.subarraytype);
    MPI_Type_commit(&parallel.subarraytype);


}

void Field::generate(const ParallelData& parallel) {

    int dims[3], coords[3], periods[3];
    MPI_Cart_get(parallel.comm, 3, dims, periods, coords);

    // Radius of the source disc 
    double radius = (nx_full + ny_full + nz_full) / 18.0;
    for (int i = 0; i < nx + 2; i++) {
        for (int j = 0; j < ny + 2; j++) {
            for (int k = 0; k < nz + 2; k++) {
                // Distance of point i, j, k from the origin 
                auto dx = i + coords[0] * nx - nx_full / 2 + 1;
                auto dy = j + coords[1] * ny - ny_full / 2 + 1;
                auto dz = k + coords[2] * nz - nz_full / 2 + 1;
                if (dx * dx + dy * dy + dz * dz < radius * radius) {
                    temperature(i, j, k) = 5.0;
                } else {
                    temperature(i, j, k) = 65.0;
                }
            }
        }
    }

    // Boundary conditions
    if (0 == coords[2])
      for (int i = 0; i < nx + 2; i++) {
        for (int j = 0; j < ny + 2; j++) {
          temperature(i, j, 0) = 20.0;
        }
      }
    if (coords[2] == dims[2] - 1)
      for (int i = 0; i < nx + 2; i++) {
        for (int j = 0; j < ny + 2; j++) {
          temperature(i, j, nz + 1) = 35.0;      
        }
      }

    if (0 == coords[1])
      for (int i = 0; i < nx + 2; i++) {
        for (int k = 0; k < nz + 2; k++) {
          temperature(i, 0, k) = 35.0;
        }
      }
    if (coords[1] == dims[1] - 1)
      for (int i = 0; i < nx + 2; i++) {
        for (int k = 0; k < nz + 2; k++) {
          temperature(i, ny + 1, k) = 20.0;
      }
    }

    if (0 == coords[0])
      for (int j = 0; j < ny + 2; j++) {
        for (int k = 0; k < nz + 2; k++) {
          temperature(0, j, k) = 20.0;
        }
      }
    if (coords[0] == dims[0] - 1)
      for (int j = 0; j < ny + 2; j++) {
        for (int k = 0; k < nz + 2; k++) {
          temperature(nx + 1, j, k) = 35.0;
        }
      }

}
