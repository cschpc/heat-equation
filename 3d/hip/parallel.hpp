#pragma once
#include <string>
#include <cstdio>
#include <mpi.h>
#include "matrix.hpp"

// Class for basic parallelization information
struct ParallelData {
    int size;            // Number of MPI tasks
    int dims[3] = {0, 0, 0};
    int rank;
    int ngbrs[3][2];     // Ranks of neighbouring MPI tasks
#ifdef MPI_DATATYPES
    MPI_Datatype halotypes[3];
#else
    Matrix<double> send_buffers[3][2];
    Matrix<double> recv_buffers[3][2];
#endif
    MPI_Datatype subarraytype;
    MPI_Request requests[12];
    MPI_Comm comm;

    ParallelData() {     // Constructor

      MPI_Comm_size(MPI_COMM_WORLD, &size);

      constexpr int ndims = 3;
      int periods[ndims] = {0, 0, 0};

      MPI_Dims_create(size, ndims, dims);
      MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm);
      MPI_Comm_rank(comm, &rank);

      // Determine neighbors
      for (int i=0; i < ndims; i++)
        MPI_Cart_shift(comm, i, 1, &ngbrs[i][0], &ngbrs[i][1]);

    };

};
