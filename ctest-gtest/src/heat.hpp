#pragma once

namespace heat {
struct Input;
}

struct Field;
struct ParallelData;

// Function declarations
Field initialize(const heat::Input &input, const ParallelData &parallel);
void exchange(Field &field, const ParallelData &parallel);
void evolve(Field& curr, const Field& prev, const double a, const double dt);

namespace heat {
double stencil(int i, int j, const Field &field, double a, double dt);
}
