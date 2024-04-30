#pragma once

#include <vector>

struct Field;
struct ParallelData;
namespace heat {
double average(const Field &field, const ParallelData &parallel);
std::vector<double> generate_field(int num_rows, int num_cols, int rank);
std::vector<double> scatter(std::vector<double> &&full_data,
                            int num_values_per_rank, int n);
}
