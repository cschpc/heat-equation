#pragma once
#ifndef NO_MPI
#include <mpi.h>
#endif
#include "matrix.hpp"
#include <Kokkos_Core.hpp>

// Class for basic parallelization information
struct ParallelData {
    int size;            // Number of MPI tasks
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};
    int rank;
    int ngbrs[3][2];     // Ranks of neighbouring MPI tasks
#ifndef NO_MPI
    Kokkos::View<double**> send_buffers[3][2];
    Kokkos::View<double**> recv_buffers[3][2];
    MPI_Request requests[12];
    MPI_Comm comm;
#endif

    ParallelData() {     // Constructor

#ifdef NO_MPI
      size = 1;
      rank = 0;
      dims[0] = dims[1] = dims[2] = 1;
#else
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      constexpr int ndims = 3;
      int periods[ndims] = {0, 0, 0};

      MPI_Dims_create(size, ndims, dims);
      MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm);
      MPI_Comm_rank(comm, &rank);
      MPI_Cart_get(comm, 3, dims, periods, coords);

      // Determine neighbors
      for (int i=0; i < ndims; i++)
        MPI_Cart_shift(comm, i, 1, &ngbrs[i][0], &ngbrs[i][1]);

#endif
    };

    void finalize() { // Clean-up MPI resources
      MPI_Comm_free(&comm);
    }


};

