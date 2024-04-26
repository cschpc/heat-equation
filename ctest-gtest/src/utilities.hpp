#pragma once

struct Field;
struct ParallelData;
namespace heat {
double average(const Field &field, const ParallelData &parallel);
}
