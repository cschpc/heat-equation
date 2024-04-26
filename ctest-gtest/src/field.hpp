#pragma once

#include "matrix.hpp"

struct ParallelData;

// Class for temperature field
struct Field {
    // num_rows and num_cols are the true dimensions of the field. The
    // temperature matrix contains also ghost layers, so it will have dimensions
    // num_rows+2 x num_cols+2
    int num_rows; // Local dimensions of the field
    int num_cols;
    int num_rows_global;        // Global dimensions of the field
    int num_cols_global;        // Global dimensions of the field

    // Grid spacing
    static constexpr double dx = 0.01;
    static constexpr double dy = 0.01;
    static constexpr double dx2 = dx * dx;
    static constexpr double dy2 = dy * dy;
    static constexpr double inv_dx2 = 1.0 / dx2;
    static constexpr double inv_dy2 = 1.0 / dy2;

    Matrix<double> temperature;

    void setup(int num_rows_in, int num_cols_in, const ParallelData &parallel);
    void generate(const ParallelData &parallel);
    // standard (i,j) syntax for setting elements
    double& operator()(int i, int j) {return temperature(i, j);}
    // standard (i,j) syntax for getting elements
    const double& operator()(int i, int j) const {return temperature(i, j);}

    static std::pair<int, int> partition_domain(int width, int height,
                                                int num_partitions);
};
