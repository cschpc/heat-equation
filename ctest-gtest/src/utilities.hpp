#pragma once

#include <vector>

struct Field;
struct ParallelData;
namespace heat {
double average(const Field &field, const ParallelData &parallel);
std::tuple<int, int, std::vector<double>> generate_field(int num_rows,
                                                         int num_cols);
std::vector<double> scatter(std::vector<double> &&full_data,
                            int num_values_per_rank, int n);
std::vector<double> gather(const Field &field, const ParallelData &parallel);
}
