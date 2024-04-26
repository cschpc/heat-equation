#pragma once
#include <string>
#include <tuple>
#include <vector>

struct Field;
struct ParallelData;

void write_field(const Field &field, const int iter,
                 const ParallelData &parallel);

void read_field(Field &field, const std::string &filename,
                const ParallelData &parallel);

namespace heat {
std::tuple<int, int, std::vector<double>>
read_field(const std::string &filename, int rank);
}
