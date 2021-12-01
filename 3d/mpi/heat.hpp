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

// Class for temperature field
struct Field {
    // nx and ny are the true dimensions of the field. The temperature matrix
    // contains also ghost layers, so it will have dimensions nx+2 x ny+2 
    int nx;                     // Local dimensions of the field
    int ny;
    int nz;
    int nx_full;                // Global dimensions of the field
    int ny_full;                // Global dimensions of the field
    int nz_full;                // Global dimensions of the field
    double dx = 0.01;           // Grid spacing
    double dy = 0.01;
    double dz = 0.01;

    Matrix<double> temperature;

    void setup(int nx_in, int ny_in, int nz_in, ParallelData& parallel);

    void generate(const ParallelData& parallel);

    // standard (i,j) syntax for setting elements
    double& operator()(int i, int j, int k) {return temperature(i, j, k);}

    // standard (i,j) syntax for getting elements
    const double& operator()(int i, int j, int k) const {return temperature(i, j, k);}

};

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData& parallel);

void exchange(Field& field, ParallelData& parallel);

void evolve(Field& curr, const Field& prev, const double a, const double dt);

void write_field(Field& field, const int iter, const ParallelData& parallel);

void read_field(Field& field, std::string filename,
                ParallelData& parallel);

double average(const Field& field);
