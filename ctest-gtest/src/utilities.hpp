#pragma once

#include <vector>

struct Field;
struct ParallelData;
namespace heat {
double average(const Field &field, const ParallelData &parallel);
std::vector<double> generate_field(int num_rows, int num_cols, int rank);
}
