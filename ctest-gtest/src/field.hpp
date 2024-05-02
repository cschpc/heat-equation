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

    // Grid spacing
    static constexpr double dx = 0.01;
    static constexpr double dy = 0.01;
    static constexpr double dx2 = dx * dx;
    static constexpr double dy2 = dy * dy;
    static constexpr double inv_dx2 = 1.0 / dx2;
    static constexpr double inv_dy2 = 1.0 / dy2;

  private:
    Matrix<double> temperature;

  public:
    Field(std::vector<double> &&data, int num_rows, int num_cols);
    // standard (i,j) syntax for setting elements
    double &operator()(int i, int j) { return temperature(i + 1, j + 1); }
    // standard (i,j) syntax for getting elements
    const double &operator()(int i, int j) const {
        return temperature(i + 1, j + 1);
    }
    double sum() const;
    std::vector<double> get_data() const;
    static std::pair<int, int> partition_domain(int num_rows, int num_cols,
                                                int num_partitions);
    // This is somewhat misleading...
    double *data(int i = 0, int j = 0) { return temperature.data(i, j); }
};
