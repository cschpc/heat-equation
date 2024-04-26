#pragma once
#include "field.hpp"
#include <string>

namespace heat {
struct Input;
}

// Function declarations
Field initialize(const heat::Input &input, const ParallelData &parallel);

void exchange(Field &field, const ParallelData &parallel);

void evolve(Field& curr, const Field& prev, const double a, const double dt);

void write_field(const Field &field, const int iter,
                 const ParallelData &parallel);

void read_field(Field &field, const std::string &filename,
                const ParallelData &parallel);

double average(const Field &field, const ParallelData &parallel);

namespace heat {
double stencil(int i, int j, const Field &field, double a, double dt);
}
