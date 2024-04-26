#pragma once
#include <string>
#include "matrix.hpp"

namespace heat {
struct Input;
}

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
    int nx_full;                // Global dimensions of the field
    int ny_full;                // Global dimensions of the field

    // Grid spacing
    static constexpr double dx = 0.01;
    static constexpr double dy = 0.01;
    static constexpr double dx2 = dx * dx;
    static constexpr double dy2 = dy * dy;
    static constexpr double inv_dx2 = 1.0 / dx2;
    static constexpr double inv_dy2 = 1.0 / dy2;

    Matrix<double> temperature;

    void setup(int nx_in, int ny_in, const ParallelData &parallel);

    void generate(const ParallelData &parallel);

    // standard (i,j) syntax for setting elements
    double& operator()(int i, int j) {return temperature(i, j);}

    // standard (i,j) syntax for getting elements
    const double& operator()(int i, int j) const {return temperature(i, j);}

};

// Function declarations
Field initialize(const heat::Input &input, const ParallelData &parallel);

void exchange(Field &field, const ParallelData &parallel);

void evolve(Field& curr, const Field& prev, const double a, const double dt);

void write_field(const Field &field, const int iter,
                 const ParallelData &parallel);

void read_field(Field &field, const std::string &filename,
                const ParallelData &parallel);

double average(const Field& field);

namespace heat {
double stencil(int i, int j, const Field &field, double a, double dt);
}
