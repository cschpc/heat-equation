#pragma once
#ifndef NO_MPI
#include <mpi.h>
#endif
#include "matrix.hpp"
#ifdef _OPENMP
#include <cstdio>
#include <omp.h>
#endif

// Class for basic parallelization information
struct ParallelData {
    int size;            // Number of MPI tasks
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};
    int rank;
    int num_threads = 1;
    int thread_id = 0;
    int ngbrs[3][2];     // Ranks of neighbouring MPI tasks
#ifndef NO_MPI
#ifdef MPI_DATATYPES
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
#ifdef _OPENMP
      int tmp_threads;  // class members do not work nicely with OpenMP; use tmp variable
#pragma omp parallel
#pragma omp single
      tmp_threads = omp_get_num_threads();
      num_threads = tmp_threads;      
#endif
    };
};

