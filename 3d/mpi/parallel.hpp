#pragma once
#ifndef NO_MPI
#include <mpi.h>
#endif
#include "matrix.hpp"

// Class for basic parallelization information
struct ParallelData {
    int size;            // Number of MPI tasks
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};
    int rank;
    int ngbrs[3][2];     // Ranks of neighbouring MPI tasks
#ifndef NO_MPI
#if defined MPI_DATATYPES || defined MPI_NEIGHBORHOOD || defined MPI_ONESIDED
    MPI_Datatype halotypes[3];
#else
    Matrix<double> send_buffers[3][2];
    Matrix<double> recv_buffers[3][2];
#endif
    MPI_Datatype subarraytype;
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
#if defined MPI_DATATYPES || defined MPI_NEIGHBORHOOD
      for (int i=0; i < 3; i++)
         MPI_Type_free(&halotypes[i]);
#endif
      MPI_Type_free(&subarraytype);
      MPI_Comm_free(&comm);
    }


};

