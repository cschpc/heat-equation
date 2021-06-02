#pragma once
#include <string>
#include "matrix.hpp"

// Class for basic parallelization information
struct ParallelData {
    int size;            // Number of MPI tasks
    int rank;
    int nup, ndown;      // Ranks of neighbouring MPI tasks

    ParallelData();      // Constructor
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

    void setup(int nx_in, int ny_in, int nz_in, ParallelData parallel);

    void generate(ParallelData parallel);

    // standard (i,j) syntax for setting elements
    double& operator()(int i, int j, int k) {return temperature(i, j, k);}

    // standard (i,j) syntax for getting elements
    const double& operator()(int i, int j, int k) const {return temperature(i, j, k);}

};

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData parallel);

void exchange(Field& field, const ParallelData parallel);

void evolve(Field& curr, const Field& prev, const double a, const double dt);

void write_field(const Field& field, const int iter, const ParallelData parallel);

void read_field(Field& field, std::string filename,
                const ParallelData parallel);

double average(const Field& field);
